import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import os

def get_cora_dataset():
    """
    Downloads/Loads the Cora dataset from the 'data' folder.
    """
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    dataset = Planetoid(root='data/Planetoid', name='Cora') # Removed NormalizeFeatures for Fuzzy Logic compatibility
    return dataset

if __name__ == "__main__":
    # Test the loader
    dataset = get_cora_dataset()
    data = dataset[0]
    print("--- SUCCESS ---")
    print(f"Dataset Loaded: {dataset.name}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Features: {data.num_features}")
    print(f"Classes: {dataset.num_classes}")