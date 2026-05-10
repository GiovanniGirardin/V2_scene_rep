Project structure:

scene_rep_smarts/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── smarts_left_turn.yaml
│   ├── smarts_roundabout.yaml
│   └── smarts_double_merge.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── collect_random_rollouts.py
├── scene_rep/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── smarts_env.py              # SMARTS wrapper
│   │   ├── action_adapter.py          # speed + lane-change action mapping
│   │   ├── observation_adapter.py     # SMARTS obs -> M_t, K_t
│   │   └── reward.py
│   ├── data/
│   │   ├── replay_buffer.py
│   │   ├── future_queue.py            # for SLT future sequences
│   │   └── batch.py
│   ├── models/
│   │   ├── common.py                  # MLP, masks, normalization
│   │   ├── transformer_blocks.py
│   │   ├── mst_encoder.py             # Multi-Stage Transformer
│   │   ├── slt.py                     # Sequential Latent Transformer
│   │   ├── actor.py
│   │   ├── critic.py
│   │   └── sac.py
│   ├── training/
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── augmentation.py            # ego-centric transform + random rotation
│   │   └── checkpointing.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── seed.py
│   │   └── geometry.py
│   └── evaluation/
│       ├── metrics.py                 # success, collision, stagnation
│       └── rollout.py
└── tests/
    ├── test_shapes.py
    ├── test_replay_buffer.py
    └── test_mst_forward.py






Framework: PyTorch, not TensorFlow
Simulator: SMARTS only
RL algorithm: SAC
Scene encoder: MST
Auxiliary training module: SLT
Observation format:
  M_t: [num_agents, history_len, 5]
       x, y, vx, vy, heading
  K_t: [num_agents, num_candidate_routes, waypoint_len, 3]
       waypoint_x, waypoint_y, waypoint_heading
Action:
  [target_speed_normalized, lane_change_continuous]
  lane_change_continuous later discretized into left / keep / right


