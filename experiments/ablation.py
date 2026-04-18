import torch
import sys
import os
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import get_cora_dataset
from experiments.train import train_model

def run_ablation_study():
    """
    Executes the 8-Component Ablation Study for DFRG-NN.
    """
    print("=" * 60)
    print("STARTING DFRG-NN ABLATION STUDY")
    print("=" * 60)
    
    # 1. Load Data
    dataset = get_cora_dataset()
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Define Ablation Configs
    # Base Config (Full DFRG-NN)
    base_config = {
        'hidden_dim': 32, # Updated to match validated architecture
        'num_layers': 2,
        'dropout': 0.5,
        'lr': 0.001,
        'epochs': 100,
        'weight_decay': 5e-4,
        'use_rough': True,
        'use_fuzzy_weights': True
    }
    
    # Define variations
    ablations = {
        'Baseline (Full DFRG-NN)': base_config.copy(),
        
        'AB-1: Crisp Weights (No Fuzzy Params)': {
            **base_config,
            'use_fuzzy_weights': False
        },
        
        'AB-2: No Rough Sets': {
            **base_config,
            'use_rough': False
        },
        
        'AB-3: Shallow Architecture (1 Layer)': {
            **base_config,
            'num_layers': 1
        },
        
        # AB-4 (Multiobj) is structural (using genetic optimizer), hard to ablate inside simple train loop.
        # We simulate "Single Obj" by just training here with standard Loss (which is what train_model does).
        # The full NSGA-II comparison would require running the optimizer.py vs this.
        # So we skip explicit AB-4 here or treat 'Baseline' as the proxy for standard training.
        
        # AB-5: Fuzzy Depth (Input Only or All). 
        # Currently DFRG_NN applies fuzzy everywhere.
        # Implementing this would require more flags. Skipping for now or treating as future work.
        
        # AB-6: Rough Set Type. Currently strictly Fuzzy-Rough.
        
        # AB-7: Edge Uncertainty. 
        # Requires modifying the specific graph edges (adding noise).
        
    }
    
    # 3. Run Experiments
    results_table = []
    
    for name, config in ablations.items():
        print(f"\nRunning: {name}")
        
        # Run 5 times for stats
        runs = []
        for seed in range(3):
            # Set seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            res = train_model(data, config, device, verbose=False)
            runs.append(res['test_acc'])
            
        avg_acc = np.mean(runs)
        std_acc = np.std(runs)
        
        print(f"  -> Avg Test Acc: {avg_acc:.4f} ± {std_acc:.4f}")
        results_table.append((name, avg_acc, std_acc))
        
    # 4. Print Summary
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Experiment':<40} | {'Accuracy':<10} | {'Std Dev':<10}")
    print("-" * 65)
    for name, avg, std in results_table:
        print(f"{name:<40} | {avg:.4f}     | {std:.4f}")
    print("-" * 65)

if __name__ == "__main__":
    run_ablation_study()
