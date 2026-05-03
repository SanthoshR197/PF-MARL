import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_paper_graphs():
    os.makedirs("results/plots", exist_ok=True)
    
    files_to_check = {
        "Centralized MADDPG": "results/centralized_metrics.csv",
        "FedMADDPG (No DP)": "results/fl_client_no_dp_metrics.csv",
        "PF-MARL (FL + DP)": "results/fl_client_default_metrics.csv"
    }
    
    data = {}
    for label, path in files_to_check.items():
        if os.path.exists(path):
            try:
                # Read CSV
                df = pd.read_csv(path)
                
                # Apply moving average for smoothing
                window_size = max(1, len(df) // 20)  # Dynamic window size
                
                if 'Total_Reward' in df.columns:
                    # Centralized format
                    data[label] = {
                        'reward': df['Total_Reward'].rolling(window=window_size).mean(),
                        'infected': df['Total_Infected'].rolling(window=window_size).mean()
                    }
                else:
                    # FL format
                    data[label] = {
                        'reward': df['reward'].rolling(window=window_size).mean(),
                        'infected': df['infected'].rolling(window=window_size).mean()
                    }
            except Exception as e:
                print(f"Error reading {path}: {e}")
        else:
            print(f"Warning: {path} not found. Ensure the baseline was run.")
            
    if not data:
        print("No data found to plot!")
        return
        
    # --- Plotting Total Reward ---
    plt.figure(figsize=(10, 6))
    for label, metrics in data.items():
        if 'reward' in metrics:
            plt.plot(metrics['reward'], label=label)
    
    plt.title("Total Evaluated Reward (Higher is Better)")
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/reward_comparison.png", dpi=300)
    print("Saved reward comparison to results/plots/reward_comparison.png")
    
    # --- Plotting Infection Count ---
    plt.figure(figsize=(10, 6))
    for label, metrics in data.items():
        if 'infected' in metrics:
            plt.plot(metrics['infected'], label=label)
            
    # Add Static Baseline References (Based on benchmark evaluation results)
    plt.axhline(y=358.2, color='r', linestyle='--', label='No-Action Baseline', alpha=0.5)
    plt.axhline(y=266.5, color='orange', linestyle='--', label='Random Baseline', alpha=0.5)
    plt.axhline(y=40.0, color='g', linestyle='--', label='Total Lockdown', alpha=0.5)

    plt.title("Total Infections per Episode (Lower is Better)")
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Infections")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plots/infection_comparison.png", dpi=300)
    print("Saved infection comparison to results/plots/infection_comparison.png")
    
    print("Plots updated with baseline markers! Ready for the paper.")

if __name__ == "__main__":
    plot_paper_graphs()
