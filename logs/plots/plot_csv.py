import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CSV_PATH = "logs/training_episodes.csv"
OUT_DIR = Path("logs/plots")

START_ROW = 0
ROLLING_WINDOW = 100
MAX_TRAINING_STEP = 100000

OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# Usa solo le righe dalla 1398 in poi (attenzione: indice 0-based)
df = df.iloc[START_ROW:].reset_index(drop=True)

# Rimuove eventuali duplicati
df = df.drop_duplicates(subset=["global_step", "episode"])

if df.empty:
    raise ValueError(f"Nessuna riga disponibile dopo START_ROW={START_ROW}")

x = df["global_step"]


# === METRICHE ===
success_rate = df["success"].mean()
collision_rate = df["collision"].mean()
off_route_rate = df["off_route"].mean()
timeout_rate = df["timeout"].mean()

print(f"=== METRICHE GLOBALI (da riga {START_ROW} in poi) ===")
print(f"Success rate: {success_rate:.3f}")
print(f"Collision rate: {collision_rate:.3f}")
print(f"Off-route rate: {off_route_rate:.3f}")
print(f"Timeout rate: {timeout_rate:.3f}")
print(f"Training steps: {int(df['global_step'].min())} -> {int(df['global_step'].max())}")


# === PLOT 1: Episode Return ===
plt.figure(figsize=(12, 5))
plt.plot(x, df["episode_return"], alpha=0.7)
plt.title("Episode Return over Training (filtered)")
plt.xlabel("Training step")
plt.ylabel("Return")
plt.xlim(0, MAX_TRAINING_STEP)
plt.grid(True)
save_plot("training_episode_return.png")


# === PLOT 2: Episode Length ===
plt.figure(figsize=(12, 5))
plt.plot(x, df["episode_length"], alpha=0.7, color="orange")
plt.title("Episode Length over Training (filtered)")
plt.xlabel("Training step")
plt.ylabel("Length")
plt.xlim(0, MAX_TRAINING_STEP)
plt.grid(True)
save_plot("training_episode_length.png")


# === PLOT 3: Terminal Reason Distribution ===
plt.figure(figsize=(10, 5))
sns.countplot(x="terminal_reason", data=df)
plt.title("Distribution of Terminal Reasons (filtered)")
plt.xticks(rotation=45)
plt.grid(True, axis="y")
save_plot("training_terminal_reasons.png")


# === PLOT 4: Rolling Average Return ===
df["rolling_return"] = df["episode_return"].rolling(window=ROLLING_WINDOW).mean()

plt.figure(figsize=(12, 5))
plt.plot(x, df["rolling_return"], color="green")
plt.title(f"Rolling Average Episode Return (window={ROLLING_WINDOW}, filtered)")
plt.xlabel("Training step")
plt.ylabel("Rolling Return")
plt.xlim(0, MAX_TRAINING_STEP)
plt.grid(True)
save_plot("training_rolling_return.png")


# === PLOT 5: Rolling Failure Rates ===
df["rolling_collision"] = df["collision"].rolling(window=ROLLING_WINDOW).mean()
df["rolling_off_route"] = df["off_route"].rolling(window=ROLLING_WINDOW).mean()
df["rolling_timeout"] = df["timeout"].rolling(window=ROLLING_WINDOW).mean()

plt.figure(figsize=(12, 5))
plt.plot(x, df["rolling_collision"], label="Collision", color="red")
plt.plot(x, df["rolling_off_route"], label="Off-route", color="purple")
plt.plot(x, df["rolling_timeout"], label="Timeout", color="blue")
plt.title(f"Rolling Failure Rates (window={ROLLING_WINDOW}, filtered)")
plt.xlabel("Training step")
plt.ylabel("Rate")
plt.xlim(0, MAX_TRAINING_STEP)
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
save_plot("training_rolling_failure_rates.png")
