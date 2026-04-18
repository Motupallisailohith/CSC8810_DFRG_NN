import torch
import torch.nn.functional as F
import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import DFRG_NN
from experiments.metrics import calculate_metrics, get_uncertainty_score

def train_model(data, config, device, verbose=False):
    """
    Trains a DFRG-NN model with specific configuration.
    
    Args:
        data: PyG Data object
        config: Dictionary of hyperparameters:
                {
                    'hidden_dim': int,
                    'num_layers': int,
                    'dropout': float,
                    'lr': float,
                    'weight_decay': float,
                    'epochs': int
                }
        device: torch.device
        verbose: bool, print progress
        
    Returns:
        metrics_dict: Dictionary containing final performance metrics
    """
    
    # 1. Initialize Model
    model = DFRG_NN(
        num_features=data.num_features,
        num_classes=data.y.max().item() + 1,
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.5),
        use_rough=config.get('use_rough', True),
        use_fuzzy_weights=config.get('use_fuzzy_weights', True)
    ).to(device)
    
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.01), weight_decay=config.get('weight_decay', 5e-4))

    # 2. Training Loop
    best_val_acc = 0.0
    final_test_acc = 0.0
    final_complexity = 0
    final_uncertainty = 0.0
    
    history = {'loss': [], 'val_acc': [], 'train_acc': []}
    
    for epoch in range(config.get('epochs', 200)):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out, uncertainties = model(data.x, data.edge_index)
        
        # Loss: NLL Loss + Uncertainty Regularization (Optional)
        # We can penalize high uncertainty to force the model to be confident
        nll_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        uncertainty_loss = get_uncertainty_score(uncertainties) * 0.0 # Temporarily disable to debug stability
        
        loss = nll_loss + uncertainty_loss
        
        loss.backward()
        optimizer.step()
        
        # Record History
        model.eval()
        with torch.no_grad():
             # Quick train acc check (expensive to do every epoch but needed for nice plots)
             # To save time, we can assume 'out' from training is close enough, but for accuracy we re-eval
             # Let's just use the training out directly to approximate classification accuracy
             train_pred = out.argmax(dim=1)
             train_correct = (train_pred[data.train_mask] == data.y[data.train_mask]).sum()
             train_acc = int(train_correct) / int(data.train_mask.sum())
             
             history['loss'].append(loss.item())
             history['train_acc'].append(train_acc)

        # Validation
        if epoch % 10 == 0 or epoch == config.get('epochs', 200) - 1:
            model.eval()
            with torch.no_grad():
                val_out, _ = model(data.x, data.edge_index)
                val_metrics = calculate_metrics(val_out, data.y, data.val_mask)
                val_acc = val_metrics['Accuracy']
                
                history['val_acc'].append((epoch, val_acc))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Check test accuracy at this best validation point
                    test_metrics = calculate_metrics(val_out, data.y, data.test_mask)
                    final_test_acc = test_metrics['Accuracy']
                    
            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch:03d}: Loss {loss.item():.4f}, Val Acc {val_acc:.4f}")

    # 3. Final Evaluation
    model.eval()
    with torch.no_grad():
        out, uncertainties = model(data.x, data.edge_index)
        
        # Calculate full suite of metrics
        train_metrics = calculate_metrics(out, data.y, data.train_mask)
        val_metrics = calculate_metrics(out, data.y, data.val_mask)
        test_metrics = calculate_metrics(out, data.y, data.test_mask)
        
        # Complexity (Parameter count)
        params = sum(p.numel() for p in model.parameters())
        
        # Average global uncertainty
        avg_unc = get_uncertainty_score(uncertainties)

    results = {
        'train_acc': train_metrics['Accuracy'],
        'val_acc': val_metrics['Accuracy'],
        'test_acc': test_metrics['Accuracy'],
        'train_f1_macro': train_metrics['Macro-F1'],
        'test_f1_macro': test_metrics['Macro-F1'],
        'uncertainty': avg_unc,
        'complexity': params,
        'generalization_gap': abs(train_metrics['Accuracy'] - test_metrics['Accuracy']),
        'history': history # Return history for plotting
    }
    
    return results
