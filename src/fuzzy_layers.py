import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class FuzzificationLayer(nn.Module):
    """
    Converts crisp input features into fuzzy membership values using Gaussian membership functions.
    Expands input dimension by factor of num_fuzzy_sets (e.g., Low, Medium, High).
    """
    def __init__(self, in_features, num_fuzzy_sets=3):
        super(FuzzificationLayer, self).__init__()
        self.num_fuzzy_sets = num_fuzzy_sets
        self.in_features = in_features
        
        # Learnable parameters: centers (mu) and widths (sigma) for the Gaussians
        # Initialize centers to cover the range [0, 1] roughly if features are normalized
        # Initialize centers with linspace 0 to 1 for each feature
        # Shape: (in_features, num_fuzzy_sets)
        centers = torch.linspace(0, 1, num_fuzzy_sets).unsqueeze(0).repeat(in_features, 1)
        self.centers = nn.Parameter(centers)
        self.sigmas = nn.Parameter(torch.ones(in_features, num_fuzzy_sets) * 0.2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, F)
        Returns:
            Fuzzy memberships of shape (N, F * num_fuzzy_sets)
        """
        # Expand x to (N, F, 1) to broadcast against parameters (F, K)
        x_expanded = x.unsqueeze(-1) 
        
        # Ensure positive sigmas
        sigmas = torch.abs(self.sigmas) + 1e-5
        
        # Gaussian Membership Function: exp( - (x - c)^2 / (2 * s^2) )
        num = (x_expanded - self.centers).pow(2)
        den = 2 * sigmas.pow(2)
        memberships = torch.exp(-num / den) # Shape: (N, F, K)
        
        # Flatten to (N, F * K)
        return memberships.reshape(x.size(0), -1)

class FuzzyGraphConvolution(MessagePassing):
    """
    Graph Convolutional Layer with Fuzzy Parameters (Weights are Gaussian Fuzzy Numbers).
    Propagates both Mean (Mu) and Uncertainty (Sigma).
    """
    def __init__(self, in_channels, out_channels):
        super(FuzzyGraphConvolution, self).__init__(aggr='add') 
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Fuzzy Weights: Mean (mu) and Uncertainty/Width (sigma)
        self.weight_mu = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_sigma = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        # Bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_sigma, 1e-4) # Start with VERY low uncertainty (almost crisp)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_sigma, 1e-4)

    def forward(self, x_mu, edge_index, x_sigma=None):
        """
        Args:
            x_mu: Input means (N, in_channels)
            edge_index: Graph connectivity
            x_sigma: Input uncertainties (N, in_channels). If None, assumed 0 (crisp).
        Returns:
            out_mu, out_sigma
        """
        if x_sigma is None:
            x_sigma = torch.zeros_like(x_mu)

        # 1. Fuzzy Matrix Multiplication (Linear Transformation)
        # Approximate arithmetic for Gaussian Fuzzy Numbers:
        # Mean_out = Mean_in * Mean_weight
        # Sigma_out approx |Mean_in|*Sigma_weight + Sigma_in*|Mean_weight|
        
        # Linear projection
        proj_mu = torch.matmul(x_mu, self.weight_mu)
        proj_sigma = torch.matmul(torch.abs(x_mu), torch.abs(self.weight_sigma)) + \
                     torch.matmul(x_sigma, torch.abs(self.weight_mu))

        # Add Bias
        proj_mu = proj_mu + self.bias_mu
        proj_sigma = proj_sigma + self.bias_sigma

        # 2. Graph Propagation
        # Calculate normalization
        row, col = edge_index
        deg = degree(col, x_mu.size(0), dtype=x_mu.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate
        out_mu = self.propagate(edge_index, x=proj_mu, norm=norm)
        out_sigma = self.propagate(edge_index, x=proj_sigma, norm=norm)
        
        return out_mu, out_sigma

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class DefuzzificationLayer(nn.Module):
    """
    Converts Fuzzy Outputs (Mu, Sigma) back to Crisp Values.
    Uses a weighted average based on uncertainty: prefer values with lower sigma (higher operational confidence)
    or simply returns Mu (Center of Gravity approximation for symmetric Gaussians).
    """
    def __init__(self, mode='cog'):
        super(DefuzzificationLayer, self).__init__()
        self.mode = mode

    def forward(self, x_mu, x_sigma=None):
        # Center of Gravity for Gaussian is just the Mean
        if self.mode == 'cog':
            return x_mu
        
        # If we want to penalize uncertainty:
        if self.mode == 'uncertainty_weighted':
            if x_sigma is None: return x_mu
            return x_mu / (1 + x_sigma)
            
        return x_mu
