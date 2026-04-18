import numpy as np
import torch
import sys
import os

# Add project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from experiments.train import train_model
from data_loader import get_cora_dataset

class DFRG_OptimizationProblem(ElementwiseProblem):
    """
    Pymoo Problem Definition for DFRG-NN Architecture Search.
    
    Objectives (Minimize all):
    1. Validation Error (1 - Val_Acc) -> Maximizes Accuracy
    2. Model Complexity (Normalized Parameter Count)
    3. Generalization Gap (Train_Acc - Test_Acc)
    """
    def __init__(self, data, device):
        # Define search space bounds
        # x0: num_layers [1, 5]
        # x1: hidden_dim_idx [0, 3] -> [16, 32, 64, 128]
        # x2: dropout [0.0, 0.7]
        # x3: lr_idx [0, 2] -> [0.001, 0.005, 0.01]
        # x4: use_rough [0, 1] -> Round to boolean
        # x5: use_fuzzy [0, 1] -> Round to boolean
        
        super().__init__(n_var=6,
                         n_obj=3,
                         n_ieq_constr=0,
                         xl=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                         xu=np.array([1.99, 3.99, 0.7, 2.99, 1.0, 1.0])) # Constrain layers to 1-2 since we know shallow is better
        
        self.data = data
        self.device = device
        self.hidden_dims = [16, 32, 64, 128]
        self.lrs = [0.001, 0.005, 0.01]

    def _evaluate(self, x, out, *args, **kwargs):
        # 1. Decode Genes
        num_layers = int(x[0])
        hidden_dim = self.hidden_dims[int(x[1])]
        dropout = float(x[2])
        lr = self.lrs[int(x[3])]
        use_rough = bool(round(x[4]))
        use_fuzzy = bool(round(x[5]))
        
        config = {
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'epochs': 30, # Reduced epochs for evolutionary speed
            'weight_decay': 5e-4,
            'use_rough': use_rough,
            'use_fuzzy_weights': use_fuzzy
        }
        
        # 2. Run Training
        try:
            results = train_model(self.data, config, self.device, verbose=False)
            
            # 3. Calculate Objectives
            # Obj 1: Error (Minimize)
            f1 = 1.0 - results['val_acc'] 
            
            # Obj 2: Complexity (Minimize)
            # Normalize: e.g. 50k params -> 5.0
            f2 = results['complexity'] / 10000.0 
            
            # Obj 3: Gen Gap (Minimize)
            f3 = results['generalization_gap']
            
            out["F"] = [f1, f2, f3]
            out["results"] = results # Save for tracking if needed (pymoo stores F)
            
        except Exception as e:
            print(f"Eval Failed: {e}")
            # Penalty for failure
            out["F"] = [1.0, 100.0, 1.0]

def run_nsga2_optimization(n_gen=10, pop_size=20):
    """
    Runs the NSGA-II optimization process.
    """
    print("--- Starting NSGA-II Optimization ---")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_cora_dataset()
    data = dataset[0]
    
    problem = DFRG_OptimizationProblem(data, device)
    
    # Algorithm
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.8, eta=20), # High mutation for exploration
        eliminate_duplicates=True
    )
    
    # Termination
    termination = get_termination("n_gen", n_gen)
    
    # Run
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   save_history=True,
                   verbose=True)
    
    print(f"Optimization Done. Found {len(res.X)} Pareto optimal solutions.")
    
    # --- Process and Display Results ---
    print("\n" + "="*80)
    print("PARETO OPTIMAL ARCHITECTURES (Sorted by Accuracy)")
    print("="*80)
    print(f"{'ID':<4} | {'Acc':<8} | {'Comp':<6} | {'Gap':<6} | {'Ly':<3} | {'Dim':<4} | {'Drop':<5} | {'LR':<6} | {'Rgh':<3} | {'Fuz':<3}")
    print("-" * 80)
    
    # Decode and sort by Accuracy (which is 1 - Objective[0])
    solutions = []
    for i, (f, x) in enumerate(zip(res.F, res.X)):
        acc = 1.0 - f[0]
        complexity = f[1]
        gap = f[2]
        
        # Decode Genes
        num_layers = int(x[0])
        hidden_dim = problem.hidden_dims[int(x[1])]
        dropout = x[2]
        lr = problem.lrs[int(x[3])]
        use_rough = bool(round(x[4]))
        use_fuzzy = bool(round(x[5]))
        
        sol = {
            'id': i,
            'acc': acc,
            'comp': complexity,
            'gap': gap,
            'layers': num_layers,
            'dim': hidden_dim,
            'drop': dropout,
            'lr': lr,
            'rough': use_rough,
            'fuzzy': use_fuzzy
        }
        solutions.append(sol)

    # Sort by Accuracy Descending
    solutions.sort(key=lambda s: s['acc'], reverse=True)
    
    for s in solutions[:10]: # Print top 10
        print(f"{s['id']:<4} | {s['acc']:.4f}   | {s['comp']:.2f}   | {s['gap']:.3f}  | {s['layers']:<3} | {s['dim']:<4} | {s['drop']:.2f}  | {s['lr']:<6} | {int(s['rough']):<3} | {int(s['fuzzy']):<3}")
        
    print("-" * 80)
    
    # Save to CSV
    import csv
    with open('optimization_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'acc', 'comp', 'gap', 'layers', 'dim', 'drop', 'lr', 'rough', 'fuzzy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solutions)
    print("Full results saved to 'optimization_results.csv'")

    return res

if __name__ == "__main__":
    run_nsga2_optimization()
