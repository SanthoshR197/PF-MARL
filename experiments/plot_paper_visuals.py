import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_publication_graphs():
    os.makedirs("results/plots", exist_ok=True)
    
    # 1. Load Data
    data = {}
    
    # PF-MARL
    if os.path.exists("results/fl_client_default_metrics.csv"):
        df = pd.read_csv("results/fl_client_default_metrics.csv")
        data['PF-MARL (Green)'] = {
            'inf': df['infected'].tail(50).mean(),
            'cost': df['economic_cost'].tail(50).mean()
        }
    
    # No-DP Fed
    if os.path.exists("results/fl_client_no_dp_metrics.csv"):
        df = pd.read_csv("results/fl_client_no_dp_metrics.csv")
        data['FedMADDPG (No DP)'] = {
            'inf': df['infected'].tail(50).mean(),
            'cost': df['economic_cost'].tail(50).mean()
        }
        
    # Centralized
    if os.path.exists("results/centralized_metrics.csv"):
        df = pd.read_csv("results/centralized_metrics.csv")
        inf = df['Total_Infected'].tail(50).mean()
        rew = df['Total_Reward'].tail(50).mean()
        # Back-calculate economy: Reward = - (2.0 * (inf/100)^2 + 0.5 * cost)
        # cost = (-rew - 2.0 * (inf/100)^2) / 0.5
        cost = 2 * (-rew - 0.0002 * (inf**2))
        data['Centralized MADDPG'] = {'inf': inf, 'cost': cost}

    # Static Baselines (from experimental logs)
    data['Static: No Action'] = {'inf': 244.6, 'cost': 10.0}
    data['Static: Total Lockdown'] = {'inf': 41.0, 'cost': 60.1}

    # --- GRAPH 1: Multi-Objective Pareto Analysis ---
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#7f7f7f']
    
    for i, (label, metrics) in enumerate(data.items()):
        plt.scatter(metrics['inf'], metrics['cost'], label=label, s=200, marker=markers[i], color=colors[i], edgecolors='black', zorder=5)

    # Highlight the "Optimal Direction" (Bottom Left)
    plt.annotate('Optimal Direction\n(High Health, High Economy)', xy=(40, 10), xytext=(80, 20),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2),
                 fontsize=10, fontweight='bold', ha='center')

    plt.title("Multi-Objective Pareto Analysis: Health vs. Economy", fontsize=14, fontweight='bold')
    plt.xlabel("Total Infections (Lower is Better)", fontsize=12)
    plt.ylabel("Economic Cost (Lower is Better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    
    # Add target bounds
    plt.axvline(x=244.6 * 0.7, color='r', linestyle=':', alpha=0.5, label='30% Reduction Target')
    
    plt.savefig("results/plots/pareto_analysis.png", dpi=300, bbox_inches='tight')
    print("Generated results/plots/pareto_analysis.png")

    # --- GRAPH 2: Economic Resilience Comparison Index ---
    # Convert Cost to "Activity" (100 - relative cost)
    labels = list(data.keys())
    # Normalize activity: No Action is 100%, Lockdown is the floor
    # We use (1 - cost/max_cost) index
    costs = np.array([m['cost'] for m in data.values()])
    activity_index = 100 * (1 - costs / 65.0) # Using 65 as a benchmark max cost
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, activity_index, color=colors)
    
    # Style the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.title("Economic Resilience Index (Higher = More Activity)", fontsize=14, fontweight='bold')
    plt.ylabel("% of Business-as-Usual Activity", fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 110)
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/plots/economic_resilience_index.png", dpi=300)
    print("Generated results/plots/economic_resilience_index.png")

if __name__ == "__main__":
    generate_publication_graphs()
