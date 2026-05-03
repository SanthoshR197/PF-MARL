import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def smooth(scalars, weight=0.9):
    """
    EWMA smoothing for noisy RL curves.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_learning_curves():
    centralized_path = "results/centralized_metrics.csv"
    fl_path = "results/fl_client_default_metrics.csv"
    
    if os.path.exists(fl_path):
        csv_path = fl_path
        print(f"Found Federated metrics! Plotting {csv_path}...")
    elif os.path.exists(centralized_path):
        csv_path = centralized_path
        print(f"Found Centralized metrics! Plotting {csv_path}...")
    else:
        print("Error: Could not find any metrics files in the results/ folder.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    
    # We heavily smooth the curves because MARL is extremely noisy
    smooth_weight = 0.95
    
    # Map column names depending on which simulation saved the CSV
    reward_col = 'reward' if 'reward' in df.columns else 'Total_Reward'
    infected_col = 'infected' if 'infected' in df.columns else 'Total_Infected'
    episode_col = 'Episode'
    
    if episode_col not in df.columns:
        df['Episode'] = range(1, len(df) + 1)

    df['Smoothed_Reward'] = smooth(df[reward_col].values, weight=smooth_weight)
    df['Smoothed_Infected'] = smooth(df[infected_col].values, weight=smooth_weight)

    # Set up publication styling
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Plot 1: Total Reward (Dual Objective) ---
    ax1.plot(df['Episode'], df[reward_col], alpha=0.3, color='steelblue', label='Raw Reward')
    ax1.plot(df['Episode'], df['Smoothed_Reward'], color='darkblue', linewidth=2, label='Smoothed Reward')
    ax1.set_title("MADDPG Learning Curve: Total Reward", fontweight='bold', pad=10)
    ax1.set_ylabel("Reward (Health & Economy)", fontweight='bold')
    ax1.legend(loc="lower right")
    
    # --- Plot 2: Total Infections ---
    ax2.plot(df['Episode'], df[infected_col], alpha=0.3, color='lightcoral', label='Raw Infections')
    ax2.plot(df['Episode'], df['Smoothed_Infected'], color='darkred', linewidth=2, label='Smoothed Infections')
    ax2.set_title("MADDPG Learning Curve: Epidemic Control", fontweight='bold', pad=10)
    ax2.set_xlabel("Training Episodes", fontweight='bold')
    ax2.set_ylabel("Total Infected Population", fontweight='bold')
    ax2.legend(loc="upper right")

    plt.tight_layout()
    
    # Save the plot
    output_path = "results/learning_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Publication-ready plot saved successfully to {output_path}")
    
    # Optionally display it if running in an interactive window
    # plt.show()

if __name__ == "__main__":
    plot_learning_curves()
