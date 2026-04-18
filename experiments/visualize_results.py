import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Ensure results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

def plot_pareto_front():
    """
    Reads optimization_results.csv and plots Accuracy vs Complexity (Pareto Front).
    """
    csv_path = 'optimization_results.csv'
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping Pareto plot.")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    # Color by Number of Layers, Size by Hidden Dim
    scatter = plt.scatter(df['comp'], df['acc'], 
                          c=df['layers'], 
                          s=df['dim'], 
                          cmap='viridis', 
                          alpha=0.7, 
                          edgecolors='k')
    
    plt.colorbar(scatter, label='Number of Layers')
    plt.xlabel('Model Complexity (Normalized Params)')
    plt.ylabel('Validation Accuracy')
    plt.title('Pareto Front: Accuracy vs Complexity\n(Size = Hidden Dim)')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate Top 3
    df_sorted = df.sort_values('acc', ascending=False)
    for i in range(3):
        row = df_sorted.iloc[i]
        plt.annotate(f"Rank {i+1}\n{row['acc']:.1%}", 
                     (row['comp'], row['acc']),
                     xytext=(10, -10), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", color='red'))

    output_path = 'results/pareto_front.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved Pareto Plot to {output_path}")
    plt.close()

def plot_ablation_study():
    """
    Plots the hardcoded ablation results as a bar chart.
    """
    experiments = ['Deep (3 Layers)', 'No Fuzzy', 'No Rough', 'Shallow (1 Layer)', 'Optimized (Areto)']
    accuracies = [51.7, 58.7, 65.6, 71.7, 77.4]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(experiments, accuracies, color=['gray', 'gray', 'gray', 'skyblue', 'gold'])
    
    plt.ylim(0, 100)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Ablation Study & Optimization Results')
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    output_path = 'results/ablation_study.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved Ablation Plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    try:
        plot_pareto_front()
        plot_ablation_study()
        print("Visualization Complete.")
    except Exception as e:
        print(f"Error producing plots: {e}")
