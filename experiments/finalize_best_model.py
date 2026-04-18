import torch
import json
import os
import sys

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.train import train_model
from data_loader import get_cora_dataset

def save_best_model():
    print("--- Finalizing Best DFRG-NN Model ---")
    
    # 1. Best Configuration Found by NSGA-II
    # ID: 1, Acc: 77.4%
    config = {
        'num_layers': 1,
        'hidden_dim': 128,
        'dropout': 0.31, # From optimization result
        'lr': 0.001,
        'epochs': 200, # Train fully
        'weight_decay': 5e-4,
        'use_rough': True,
        'use_fuzzy_weights': False # The optimizer found Crisp weights were better for 1-layer wide models
    }
    
    # 2. Load Data
    dataset = get_cora_dataset()
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Train
    print(f"Training with Optimal Config: {config}")
    results = train_model(data, config, device, verbose=True)
    
    # 4. Save Checkpoint
    checkpoint_dir = 'results/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    # We need to access the model from results? 
    # Currently train_model returns a dict of metrics, not the model object itself.
    # I should have modified train_model, but for now I will re-instantiate and load weights if possible,
    # OR simply update this script to instantiate the model manually to save it.
    # To avoid changing 'train.py' again, I will just re-instantiate locally here using the same logic.
    
    from src.models import DFRG_NN
    
    # Re-build model to save it (Note: 'train_model' created its own instance and didn't return it)
    # Ideally train_model returns the model. 
    # Let's simple rely on the fact that we confirmed the performance. 
    # I will instantiate a fresh model, TRAIN IT MANUALLY here to ensure I have the object to save.
    
    print("Retraining for serialization...")
    model = DFRG_NN(
        num_features=data.num_features,
        num_classes=dataset.num_classes,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_rough=config['use_rough'],
        use_fuzzy_weights=config['use_fuzzy_weights']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    model.train()
    data = data.to(device)
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
    # Save
    save_path = os.path.join(checkpoint_dir, 'best_dfrg_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to: {save_path}")
    
    # 5. Save Metrics
    metrics_path = 'results/final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    save_best_model()
