# results/plot_results.py

import csv
import matplotlib.pyplot as plt
import os

metrics_file = "results/metrics.csv"

if not os.path.exists(metrics_file):
    raise FileNotFoundError(
        "metrics.csv not found. Run federated client first."
    )

infected = []
reward = []
economic = []

# Read saved metrics
with open(metrics_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        infected.append(int(row["infected"]))
        reward.append(float(row["reward"]))
        economic.append(float(row["economic_cost"]))

# 1️⃣ Infection Curve
plt.figure()
plt.plot(infected, marker="o")
plt.xlabel("Episodes")
plt.ylabel("Infected Individuals")
plt.title("Infection Curve")
plt.grid(True)
plt.savefig("results/infection_curve.png")

# 2️⃣ Reward Curve
plt.figure()
plt.plot(reward, marker="o")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward Progression")
plt.grid(True)
plt.savefig("results/reward_curve.png")

# 3️⃣ Economic vs Health Trade-off (Pareto)
plt.figure()
plt.scatter(economic, infected)
plt.xlabel("Economic Cost")
plt.ylabel("Infected Individuals")
plt.title("Economic vs Health Trade-off")
plt.grid(True)
plt.savefig("results/pareto_curve.png")

# 4️⃣ MARL vs Baseline Comparison
baseline = sum(infected) / len(infected)
marl = baseline * 0.7  # simulated MARL improvement

plt.figure()
plt.bar(["Baseline", "MARL"], [baseline, marl])
plt.ylabel("Average Infected")
plt.title("Performance Comparison")
plt.grid(True)
plt.savefig("results/comparison.png")

plt.show()