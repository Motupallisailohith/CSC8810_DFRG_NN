import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_cora_dataset
from src.models import DFRG_NN

def explain_prediction(paper_id=0):
    print(f"\n--- EXPLAINING CLASSIFICATION FOR PAPER ID: {paper_id} ---")
    
    # 1. Load Data
    dataset = get_cora_dataset()
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Initialize Model (Using default config for demo)
    model = DFRG_NN(
        num_features=data.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=32,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    # (In a real scenario, we would load state_dict here)
    model.eval()
    
    # 3. Get Fuzzification (The "Explainable" Part)
    # We want to see which features were "High", "Medium", or "Low"
    raw_features = data.x[paper_id].unsqueeze(0).to(device) # Shape (1, 1433)
    
    with torch.no_grad():
        # A. Fuzzification Layer Output
        fuzzy_memberships = model.fuzz_layer(raw_features) # Shape (1, F*K)
        
        # Reshape to (Features, Sets)
        # Assuming 3 sets: Low, Medium, High
        num_features = data.num_features
        num_sets = 3
        fuzzy_view = fuzzy_memberships.reshape(num_features, num_sets)
        
        # B. Find Top Active Concepts
        # Sum membership across features to see global activation
        # Or find specific features that are strongly "High"
        
        # Let's find top 5 features where "High" membership is > 0.8
        high_membership = fuzzy_view[:, 2] # Index 2 is "High"
        top_values, top_indices = torch.topk(high_membership, 5)
        
        print("\n[Fuzzy Logic Trace]")
        print("The model considers this paper relevant based on these 'HIGH' features:")
        for score, idx in zip(top_values, top_indices):
            print(f"  - Feature #{idx.item()}: 'High' Membership = {score:.4f}")
            
        # C. Make Prediction
        out, _ = model(data.x.to(device), data.edge_index.to(device))
        pred_class = out[paper_id].argmax().item()
        confidence = torch.exp(out[paper_id].max()).item()
        
        print(f"\n[Final Decision]")
        print(f"  -> Predicted Class: {pred_class}")
        print(f"  -> Confidence: {confidence:.4f}")
        
        return top_indices.cpu().numpy(), fuzzy_view.cpu().numpy()

if __name__ == "__main__":
    # Test on the first paper in the dataset
    explain_prediction(paper_id=0)
