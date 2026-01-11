# =============================================================================
# UNO RL Configuration File
# =============================================================================

# Legacy parameters for original Q-learning/Monte-Carlo
params = {
    "iterations": 200,
    "algorithm": "q-learning",  # ["q-learning", "monte-carlo"]
    "logging": False,
    "model": {
        "epsilon": 0.4,
        "step_size": 0.2,
    }
}

# DQN Agent Configuration
dqn_params = {
    "epsilon": 1.0,           # Initial exploration rate
    "epsilon_decay": 0.998,   # Decay rate per episode
    "epsilon_min": 0.01,      # Minimum exploration rate
    "gamma": 0.99,            # Discount factor
    "learning_rate": 0.001,   # Neural network learning rate
    "batch_size": 64,         # Batch size for training
    "buffer_size": 20000,     # Experience replay buffer size
    "target_update_freq": 100,  # How often to update target network
    "hidden_sizes": [128, 128, 64]  # Neural network architecture
}

# Improved Q-Learning Agent Configuration
improved_qlearning_params = {
    "epsilon": 0.5,           # Initial exploration rate
    "epsilon_decay": 0.999,   # Decay rate per episode
    "epsilon_min": 0.05,      # Minimum exploration rate
    "step_size": 0.1,         # Learning rate (alpha)
    "gamma": 0.95,            # Discount factor
    "batch_size": 32,         # Batch size for experience replay
    "buffer_size": 10000      # Experience replay buffer size
}

# =============================================================================
# Stable Baselines3 Model Configurations
# =============================================================================

# Available SB3 Models (ranked by win rate)
sb3_models = {
    "selfplay_champion": {
        "path": "models/selfplay_champion.zip",
        "type": "recurrentppo",
        "description": "Self-Play Champion - 70%+ win rate (target)",
        "win_rate": 0.70
    },
    "best_recurrent_ppo": {
        "path": "models/best_recurrent_ppo_uno.zip",
        "type": "recurrentppo",
        "description": "Best Recurrent PPO - 60% win rate",
        "win_rate": 0.60
    },
    "optimal_recurrent_ppo": {
        "path": "models/optimal_recurrent_ppo.zip",
        "type": "recurrentppo",
        "description": "Optimal Recurrent PPO - 59% win rate",
        "win_rate": 0.59
    },
    "sb3_recurrent_ppo": {
        "path": "models/sb3_recurrentppo_uno.zip",
        "type": "recurrentppo",
        "description": "SB3 Recurrent PPO - 57% win rate",
        "win_rate": 0.57
    },
    "enhanced_rppo": {
        "path": "models/enhanced_rppo.zip",
        "type": "recurrentppo",
        "description": "Enhanced RPPO",
        "win_rate": 0.56
    },
    "ppo": {
        "path": "models/sb3_ppo_uno.zip",
        "type": "ppo",
        "description": "PPO Model",
        "win_rate": 0.52
    },
    "best_ppo": {
        "path": "models/best_ppo_uno.zip",
        "type": "ppo",
        "description": "Best PPO Model",
        "win_rate": 0.53
    },
    "dqn": {
        "path": "models/sb3_dqn_uno.zip",
        "type": "dqn",
        "description": "DQN Model",
        "win_rate": 0.48
    },
    "a2c": {
        "path": "models/sb3_a2c_uno.zip",
        "type": "a2c",
        "description": "A2C Model",
        "win_rate": 0.45
    },
}

# Default model to use (Self-Play Champion is the best performing model)
default_model = "selfplay_champion"

# Recurrent PPO Training Configuration (Best performing)
recurrent_ppo_config = {
    "policy": "MlpLstmPolicy",
    "learning_rate": 2.5e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "lstm_hidden_size": 128,
    "n_lstm_layers": 1,
    "total_timesteps": 500000
}

# PPO Training Configuration
ppo_config = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 200000
}

# DQN SB3 Training Configuration
dqn_sb3_config = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 100000,
    "learning_starts": 1000,
    "batch_size": 64,
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 1000,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.05,
    "total_timesteps": 200000
}

# Training Configuration
training_config = {
    "episodes": 1000,         # Number of training episodes
    "opponent": "random",     # Opponent type: "random" or "self"
    "save_freq": 100,         # Save model every N episodes
    "eval_games": 100         # Number of games for evaluation
}

# Evaluation Configuration
eval_config = {
    "num_games": 100,         # Games per evaluation
    "opponents": ["random"],  # Opponents to evaluate against
    "verbose": True
}

player_name_1 = "AI"
player_name_2 = "Human"