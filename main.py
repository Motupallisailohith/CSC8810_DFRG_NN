import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.baseline import run_baseline
from experiments.ablation import run_ablation_study
from src.optimizer import run_nsga2_optimization
from experiments.train import train_model
from data_loader import get_cora_dataset
import torch

def run_single_train():
    print("Running Single DFRG-NN Training Run...")
    dataset = get_cora_dataset()
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'hidden_dim': 64,
        'num_layers': 3,
        'dropout': 0.5,
        'lr': 0.005,
        'epochs': 50,
        'weight_decay': 5e-4,
        'use_rough': True,
        'use_fuzzy_weights': True
    }
    
    results = train_model(data, config, device, verbose=True)
    
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="CSC 8810 DFRG-NN Project Runner")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['baseline', 'train', 'ablation', 'optimize'],
                        help='Operation mode: baseline GCN, single train DFRG-NN, full ablation study, or NSGA-II optimization.')
    
    args = parser.parse_args()
    
    print(f"Executing Mode: {args.mode.upper()}")
    
    if args.mode == 'baseline':
        run_baseline()
    elif args.mode == 'train':
        run_single_train()
    elif args.mode == 'ablation':
        run_ablation_study()
    elif args.mode == 'optimize':
        run_nsga2_optimization()
    else:
        print("Invalid mode.")

if __name__ == "__main__":
    main()
