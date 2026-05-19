from __future__ import annotations

import gymnasium as gym

from smarts.core.agent_interface import AgentInterface, AgentType

SCENARIO = "/home/giovanni/SMARTS/scenarios/sumo/loop"
AGENT_ID = "Agent-007"

agent_interfaces = {
    AGENT_ID: AgentInterface.from_type(
        AgentType.Laner,
        max_episode_steps=1000,
    )
}

env = gym.make(
    "smarts.env:hiway-v1",
    scenarios=[SCENARIO],
    agent_interfaces=agent_interfaces,
    headless=True,
    seed=42,
)

obs, info = env.reset()

print("Reset OK")
print("Obs type:", type(obs))
print("Obs keys:", obs.keys() if hasattr(obs, "keys") else None)
print("Info:", info)

for i in range(5):
    action = {AGENT_ID: env.action_space[AGENT_ID].sample()}
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"step={i}")
    print("action:", action)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)

    if terminated or truncated:
        break
    
env.close()
print("SMARTS env step OK")
