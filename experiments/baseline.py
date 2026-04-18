import torch
import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_cora_dataset
from src.models import StandardGCN

def run_baseline():
    # 1. Load Data
    print("Loading data...")
    dataset = get_cora_dataset()
    data = dataset[0]
    
    # 2. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StandardGCN(dataset.num_features, dataset.num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 3. Train
    print("Training Standard GCN...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # Only calculate loss on training nodes
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # 4. Test
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    # Only evaluate on test nodes
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    
    print("-" * 30)
    print(f"FINAL BASELINE ACCURACY: {acc:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_baseline()