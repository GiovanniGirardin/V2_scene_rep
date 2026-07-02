import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "training_episodes.csv"

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)

# Usa solo le righe dalla 1398 in poi (attenzione: indice 0-based)
START_ROW = 1398
df = df.iloc[START_ROW:].reset_index(drop=True)

# Rimuove eventuali duplicati
df = df.drop_duplicates(subset=["global_step", "episode"])

# === METRICHE ===
success_rate = df["success"].mean()
collision_rate = df["collision"].mean()
off_route_rate = df["off_route"].mean()
timeout_rate = df["timeout"].mean()

print("=== METRICHE GLOBALI (da riga 1398 in poi) ===")
print(f"Success rate: {success_rate:.3f}")
print(f"Collision rate: {collision_rate:.3f}")
print(f"Off-route rate: {off_route_rate:.3f}")
print(f"Timeout rate: {timeout_rate:.3f}")

# === PLOT 1: Episode Return ===
plt.figure(figsize=(12,5))
plt.plot(df["episode"], df["episode_return"], alpha=0.7)
plt.title("Episode Return over Training (filtered)")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.show()

# === PLOT 2: Episode Length ===
plt.figure(figsize=(12,5))
plt.plot(df["episode"], df["episode_length"], alpha=0.7, color="orange")
plt.title("Episode Length over Training (filtered)")
plt.xlabel("Episode")
plt.ylabel("Length")
plt.grid(True)
plt.show()

# === PLOT 3: Terminal Reason Distribution ===
plt.figure(figsize=(10,5))
sns.countplot(x="terminal_reason", data=df)
plt.title("Distribution of Terminal Reasons (filtered)")
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.show()

# === PLOT 4: Rolling Average Return ===
df["rolling_return"] = df["episode_return"].rolling(window=50).mean()

plt.figure(figsize=(12,5))
plt.plot(df["episode"], df["rolling_return"], color="green")
plt.title("Rolling Average Episode Return (window=50, filtered)")
plt.xlabel("Episode")
plt.ylabel("Rolling Return")
plt.grid(True)
plt.show()

# === PLOT 5: Rolling Failure Rates ===
df["rolling_collision"] = df["collision"].rolling(window=50).mean()
df["rolling_off_route"] = df["off_route"].rolling(window=50).mean()
df["rolling_timeout"] = df["timeout"].rolling(window=50).mean()

plt.figure(figsize=(12,5))
plt.plot(df["episode"], df["rolling_collision"], label="Collision", color="red")
plt.plot(df["episode"], df["rolling_off_route"], label="Off-route", color="purple")
plt.plot(df["episode"], df["rolling_timeout"], label="Timeout", color="blue")
plt.title("Rolling Failure Rates (window=50, filtered)")
plt.xlabel("Episode")
plt.ylabel("Rate")
plt.legend()
plt.grid(True)
plt.show()
