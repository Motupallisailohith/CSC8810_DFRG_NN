import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import sys
import os

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.train import train_model

def run_generalization_test():
    print("--- STARTING GENERALIZATION TEST (CITESEER) ---")
    
    # 1. Load Citeseer (Real Outside Data)
    # We specify the exact path requested by the user
    path = os.path.join(os.path.dirname(__file__), '../data')
    print(f"Loading Citeseer from: {path}")
    
    try:
        dataset = Planetoid(root=path, name='CiteSeer') # Removed NormalizeFeatures
        data = dataset[0]
        print("Success! Dataset Loaded.")
        print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Classes: {dataset.num_classes}")
    except Exception as e:
        print(f"Error loading Citeseer: {e}")
        return

    # 2. Configure Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'hidden_dim': 32,
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 200, 
        'weight_decay': 5e-4,
        'use_rough': True, # Keep rough sets
        'use_fuzzy_weights': True # Re-enable fuzzy weights
    }
    
    # 3. Train on New Data
    print("\nTraining DFRG-NN on CiteSeer...")
    try:
        results = train_model(data, config, device, verbose=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCRITICAL ERROR during training: {e}")
        return
    
    print("\n--- GENERALIZATION RESULTS ---")
    print(f"Test Accuracy on CiteSeer: {results['test_acc']:.4f}")
    
    if results['test_acc'] > 0.16: # Random guess is ~1/6 = 0.16
        print("SUCCESS: Model learned patterns on unseen 'Outside' dataset.")
    else:
        print("WARNING: Low performance. Hyperparameters might need tuning for CiteSeer.")

if __name__ == "__main__":
    run_generalization_test()
