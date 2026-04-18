import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class RoughSetBlock(MessagePassing):
    """
    Implements a Fuzzy Rough Set logic block for Graph Neural Networks.
    Computes Lower and Upper approximations of node representations based on their neighborhood.
    
    Concepts:
    - Lower Approximation (Necessity): What features are common among all neighbors? (Min-pooling logic)
    - Upper Approximation (Possibility): What features exist in at least one neighbor? (Max-pooling logic)
    """
    def __init__(self, in_channels, out_channels):
        # We handle aggregation manually in forward, so aggr=None or handled per call
        super(RoughSetBlock, self).__init__(aggr=None) 
        
        # Linear transformation before approximation
        self.lin = nn.Linear(in_channels, out_channels)
        
        # Interaction layer to combine Lower and Upper
        self.combine = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index):
        # 1. Transform features
        x_emb = self.lin(x)
        
        # 2. Compute Approximations
        # Upper approx: Max feature value in neighborhood (Possibility)
        # Lower approx: Min feature value in neighborhood (Necessity)
        
        # We need to include self-loops to ensure a node is considered in its own neighborhood
        # (Reflexive property of Rough Set relations)
        # Note: edge_index should ideally include self-loops before this call, 
        # but we can assume standard message passing logic.
        
        upper = self.propagate(edge_index, x=x_emb, flow='source_to_target', aggr_method='max')
        lower = self.propagate(edge_index, x=x_emb, flow='source_to_target', aggr_method='min')
        
        # 3. Combine
        out = torch.cat([lower, upper], dim=1)
        out = self.combine(out) + x_emb # Residual connection for better gradient flow
        
        return out, lower, upper
    
    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, reduce=None, dim_size=None, aggr_method='max'):
        from torch_geometric.utils import scatter
        
        # PyG scatter syntax: scatter(src, index, dim, dim_size, reduce)
        # Note: 'reduce' arg in scatter expects 'sum', 'mul', 'mean', 'min', 'max'
        
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce=aggr_method)

class RoughSetUncertaintyQuantification(nn.Module):
    """
    Quantifies uncertainty based on the size of the Boundary Region.
    Boundary = Upper Approximation - Lower Approximation.
    Larger boundary -> Higher Uncertainty.
    """
    def __init__(self):
        super(RoughSetUncertaintyQuantification, self).__init__()

    def forward(self, lower, upper):
        """
        Returns a scalar or vector representing uncertainty.
        """
        boundary = upper - lower
        # Simple norm of boundary region vector
        uncertainty = torch.norm(boundary, p=2, dim=1, keepdim=True)
        return uncertainty
