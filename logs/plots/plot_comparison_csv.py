import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === LOAD CSV ===
csv_path = "logs/checkpoint_comparison.csv"

# Load while skipping commented lines starting with '#'
df = pd.read_csv(csv_path, comment='#')

# Sort by checkpoint step
df = df.sort_values("checkpoint_step")

# Output directory
out_dir = "plots_output"
os.makedirs(out_dir, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches='tight')
    plt.close()

# === PLOT 1: Success Rate Over Training ===
plt.figure(figsize=(10,5))
plt.plot(df["checkpoint_step"], df["success_rate"], marker='o')
plt.title("Success Rate Over Training")
plt.xlabel("Training Step")
plt.ylabel("Success Rate")
plt.grid(True)
savefig("success_rate.png")

# === PLOT 2: Failure Modes Over Training ===
plt.figure(figsize=(10,5))
plt.plot(df["checkpoint_step"], df["collision_rate"], label="Collision Rate", marker='o')
plt.plot(df["checkpoint_step"], df["off_route_rate"], label="Off-route Rate", marker='o')
plt.plot(df["checkpoint_step"], df["timeout_rate"], label="Timeout Rate", marker='o')
plt.title("Failure Modes Over Training")
plt.xlabel("Training Step")
plt.ylabel("Rate")
plt.legend()
plt.grid(True)
savefig("failure_modes.png")

# === PLOT 3: Mean Reward Over Training ===
plt.figure(figsize=(10,5))
plt.plot(df["checkpoint_step"], df["mean_reward"], marker='o')
plt.title("Mean Reward Over Training")
plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.grid(True)
savefig("mean_reward.png")

# === PLOT 4: Mean Episode Length Over Training ===
plt.figure(figsize=(10,5))
plt.plot(df["checkpoint_step"], df["mean_steps"], marker='o')
plt.title("Mean Episode Length Over Training")
plt.xlabel("Training Step")
plt.ylabel("Mean Steps")
plt.grid(True)
savefig("mean_steps.png")

# === PLOT 5: Reward Distribution Across Checkpoints ===
plt.figure(figsize=(10,5))
sns.histplot(df["mean_reward"], kde=True, bins=20)
plt.title("Distribution of Mean Reward Across Checkpoints")
plt.xlabel("Mean Reward")
plt.ylabel("Frequency")
plt.grid(True)
savefig("reward_distribution.png")

# === PLOT 6: Success vs Collision Scatter ===
plt.figure(figsize=(10,5))
plt.scatter(df["success_rate"], df["collision_rate"], c=df["checkpoint_step"], cmap="viridis", s=80)
plt.colorbar(label="Training Step")
plt.title("Success Rate vs Collision Rate")
plt.xlabel("Success Rate")
plt.ylabel("Collision Rate")
plt.grid(True)
savefig("success_vs_collision.png")
