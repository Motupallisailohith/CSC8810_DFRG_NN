import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.fuzzy_layers import FuzzificationLayer, FuzzyGraphConvolution, DefuzzificationLayer
from src.rough_sets import RoughSetBlock, RoughSetUncertaintyQuantification

class StandardGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(StandardGCN, self).__init__()
        # Layer 1: 16 hidden units
        self.conv1 = GCNConv(num_features, 16)
        # Layer 2: Output
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        # 1. Graph Conv -> ReLU -> Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # 2. Graph Conv -> Log Softmax
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class DFRG_NN(torch.nn.Module):
    """
    Deep Fuzzy Rough Graph Neural Network (DFRG-NN).
    Combines Fuzzification, Rough Set approximations, and Fuzzy Graph Convolutions.
    
    ARCHITECTURAL UPDATE:
    Uses a 'Hybrid' approach: Standard GCN for dimensionality reduction -> Fuzzy Logic for deep reasoning.
    This prevents parameter explosion on high-dimensional datasets like Cora.
    """
    def __init__(self, num_features, num_classes, hidden_dim=32, num_fuzzy_sets=3, num_layers=2, dropout=0.5,
                 use_rough=True, use_fuzzy_weights=True):
        super(DFRG_NN, self).__init__()
        self.dropout = dropout
        self.use_rough = use_rough
        self.use_fuzzy_weights = use_fuzzy_weights
        
        # 0. Pre-compression (Dimensionality Reduction)
        # 1433 features -> 32 features (Standard GCN)
        self.pre_compression = GCNConv(num_features, hidden_dim)
        
        # 1. Input Fuzzification (Applied to compacted features)
        # Takes hidden_dim (32) -> hidden_dim*K (96)
        self.fuzz_layer = FuzzificationLayer(hidden_dim, num_fuzzy_sets)
        
        fuzz_out_dim = hidden_dim * num_fuzzy_sets
        
        # 2. Stack of Blocks
        self.rough_blocks = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        # Input projection after fuzzification
        self.input_proj = nn.Linear(fuzz_out_dim, hidden_dim)
        
        for _ in range(num_layers):
            # Rough Block: computes lower/upper approx
            if self.use_rough:
                self.rough_blocks.append(RoughSetBlock(hidden_dim, hidden_dim))
            
            # Conv Layer: Fuzzy or Standard
            if self.use_fuzzy_weights:
                self.convs.append(FuzzyGraphConvolution(hidden_dim, hidden_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # 3. Uncertainty Quantification (for loss/metrics)
        self.uncertainty_quant = RoughSetUncertaintyQuantification()
        
        # 4. Defuzzification
        self.defuzz = DefuzzificationLayer(mode='uncertainty_weighted')
        
        # 5. Final Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # --- 0. Pre-compression ---
        # Reduce dimension first to avoid "Curse of Dimensionality"
        x = self.pre_compression(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # --- 1. Fuzzification ---
        # x: (N, hidden_dim)
        x_fuzz = self.fuzz_layer(x) # -> (N, hidden_dim*K)
        
        # Project to embedding space
        x_emb = self.input_proj(x_fuzz)
        x_emb = F.relu(x_emb)
        x_emb = F.dropout(x_emb, p=self.dropout, training=self.training)
        
        # Initial sigma (uncertainty) is 0 for input
        mu = x_emb
        sigma = torch.zeros_like(mu)
        
        all_layer_uncertainties = []
        
        # --- 2. Deep Stack ---
        for i in range(len(self.convs)):
            # A. Rough Set Approximation (Structural Uncertainty)
            if self.use_rough:
                rough = self.rough_blocks[i]
                mu_rough, lower, upper = rough(mu, edge_index)
                
                # Calculate structural uncertainty for this layer
                unc = self.uncertainty_quant(lower, upper)
                all_layer_uncertainties.append(unc)
            else:
                # Bypass rough block
                mu_rough = mu
            
            # B. Graph Convolution
            conv = self.convs[i]
            
            if self.use_fuzzy_weights:
                # Fuzzy Conv: propagates mu and sigma
                mu_next, sigma_next = conv(mu_rough, edge_index, sigma)
                
                mu = F.relu(mu_next)
                sigma = F.relu(sigma_next) # Sigma must be positive
                sigma = torch.clamp(sigma, max=5.0) # Prevent uncertainty explosion
                
                mu = F.dropout(mu, p=self.dropout, training=self.training)
                sigma = F.dropout(sigma, p=self.dropout, training=self.training)
            else:
                # Standard Conv: propagates only mu
                mu_next = conv(mu_rough, edge_index)
                mu = F.relu(mu_next)
                mu = F.dropout(mu, p=self.dropout, training=self.training)
                # Sigma stays 0
                sigma = torch.zeros_like(mu)

        # --- 3. Defuzzification ---
        # Combine Mu and Sigma to get crisp representation
        if self.use_fuzzy_weights:
            out_crisp = self.defuzz(mu, sigma)
        else:
            out_crisp = mu
        
        # --- 4. Classification ---
        out = self.classifier(out_crisp)
        
        return F.log_softmax(out, dim=1), all_layer_uncertainties