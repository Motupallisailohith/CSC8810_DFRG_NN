import torch
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def calculate_accuracy(pred, target):
    """
    Calculates simple accuracy: Correct / Total
    """
    correct = (pred == target).sum().item()
    return correct / target.size(0)

def calculate_metrics(pred_logits, target_labels, mask=None):
    """
    Calculates comprehensive metrics: Accuracy, Macro-F1, Micro-F1.
    Args:
        pred_logits: Raw output from model (N, C)
        target_labels: Ground truth labels (N)
        mask: Optional boolean mask (N) to filter nodes (e.g., test mask)
    Returns:
        Dictionary of metrics
    """
    if mask is not None:
        pred_logits = pred_logits[mask]
        target_labels = target_labels[mask]
        
    pred_classes = pred_logits.argmax(dim=1)
    
    # Detach for cpu/numpy
    y_true = target_labels.cpu().numpy()
    y_pred = pred_classes.cpu().numpy()
    
    acc = calculate_accuracy(pred_classes, target_labels)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    return {
        'Accuracy': acc,
        'Macro-F1': macro_f1,
        'Micro-F1': micro_f1
    }

def calculate_interpretability(model, activations=None):
    """
    Placeholder for interpretability metrics.
    In a real scenario, this would trace fuzzy rule activations.
    For now, we can estimate complexity by model parameter count or layer uncertainty.
    """
    # 1. Parameter Count (Simplicity Proxy)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. Rule Estimate (Conceptual)
    # If using Fuzzy Layers, we can check how many membership functions are "active" (non-zero)
    # This requires hooking into the layer forward pass, which is complex.
    # We will return the parameter count as a complexity proxy for optimization.
    
    return {
        'Num_Params': total_params,
        'Est_Rules': total_params // 100 # Rough proxy
    }

def calculate_generalization_gap(train_acc, test_acc):
    """
    Calculates overfitting gap. Lower is better.
    """
    return abs(train_acc - test_acc)

def get_uncertainty_score(layer_uncertainties):
    """
    Aggregates uncertainty scores from all layers.
    Args:
        layer_uncertainties: List of tensors from DFRG_NN forward pass
    """
    if not layer_uncertainties:
        return 0.0
    
    # Mean uncertainty across all layers and nodes
    total_unc = 0
    for u in layer_uncertainties:
        total_unc += u.mean().item()
        
    return total_unc / len(layer_uncertainties)
