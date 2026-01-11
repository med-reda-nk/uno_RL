"""
Train a Recurrent PPO (LSTM-based) model for UNO.
Uses sb3-contrib's RecurrentPPO for sequence-aware decision making.

Recurrent PPO is beneficial for UNO because:
- It can remember previously played cards
- It can track opponent behavior patterns
- It can make strategic decisions based on game history
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from src.sb3_agent import UnoEnv
    RPPO_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Install required packages with:")
    print("  pip install sb3-contrib stable-baselines3[extra] gymnasium")
    RPPO_AVAILABLE = False
    sys.exit(1)


class RecurrentRewardCallback(BaseCallback):
    """Custom callback for logging Recurrent PPO training metrics."""
    
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.total_games = 0
        
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[i]
                    self.total_games += 1
                    if info.get("winner") == "player":
                        self.wins += 1
        
        if self.n_calls % self.log_freq == 0:
            win_rate = (self.wins / self.total_games * 100) if self.total_games > 0 else 0
            if self.verbose > 0:
                print(f"Step {self.n_calls:,}: Games={self.total_games}, Wins={self.wins}, WinRate={win_rate:.1f}%")
        
        return True


def make_env(rank, seed=0):
    """Create a wrapped, monitored UnoEnv."""
    def _init():
        env = UnoEnv()
        env = Monitor(env)
        return env
    set_random_seed(seed + rank)
    return _init


def train_recurrent_ppo(
    total_timesteps: int = 500000,
    n_envs: int = 4,
    save_path: str = "models/recurrent_ppo_uno",
    seed: int = 42,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
):
    """
    Train a Recurrent PPO model with LSTM for UNO.
    
    The LSTM architecture allows the model to:
    - Remember sequence of moves
    - Track game state history
    - Make decisions based on temporal patterns
    
    Parameters:
    -----------
    total_timesteps : int
        Total environment steps to train
    n_envs : int
        Number of parallel environments
    save_path : str
        Path to save the trained model
    seed : int
        Random seed for reproducibility
    lstm_hidden_size : int
        Size of LSTM hidden state
    n_lstm_layers : int
        Number of LSTM layers
    """
    
    print(f"\n{'='*70}")
    print("🧠 Training Recurrent PPO (LSTM) Model for UNO")
    print(f"{'='*70}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    print(f"LSTM Hidden Size: {lstm_hidden_size}")
    print(f"LSTM Layers: {n_lstm_layers}")
    print(f"Save Path: {save_path}")
    print(f"Random Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Create vectorized environments for faster training
    print("Creating parallel environments...")
    env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(i, seed + 100) for i in range(1)])
    
    # Recurrent PPO Configuration
    rppo_config = {
        # Learning parameters
        "learning_rate": 3e-4,
        "n_steps": 128,  # Shorter rollouts work better with LSTM
        "batch_size": 128,  # Must be divisible by n_steps * n_envs
        "n_epochs": 10,
        
        # Discount and advantage estimation
        "gamma": 0.99,
        "gae_lambda": 0.95,
        
        # PPO-specific parameters
        "clip_range": 0.2,
        "normalize_advantage": True,
        
        # Entropy for exploration
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        
        # Gradient clipping for stability
        "max_grad_norm": 0.5,
        
        # LSTM Policy configuration
        "policy_kwargs": dict(
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=False,  # Separate LSTMs for policy and value
            enable_critic_lstm=True,  # LSTM for value function too
            net_arch=dict(
                pi=[128, 64],  # Policy network after LSTM
                vf=[128, 64],  # Value network after LSTM
            ),
        ),
        
        # Other settings
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": "./logs/recurrent_ppo/",
    }
    
    print("Recurrent PPO Configuration:")
    for key, value in rppo_config.items():
        if key != "policy_kwargs":
            print(f"  {key}: {value}")
    print(f"  policy_kwargs: {rppo_config['policy_kwargs']}")
    print()
    
    # Create Recurrent PPO model
    print("Initializing Recurrent PPO model with LSTM policy...")
    model = RecurrentPPO("MlpLstmPolicy", env, **rppo_config)
    
    # Print model summary
    print(f"\nModel Architecture:")
    print(f"  Policy: MlpLstmPolicy")
    print(f"  LSTM Hidden Size: {lstm_hidden_size}")
    print(f"  LSTM Layers: {n_lstm_layers}")
    print()
    
    # Callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // n_envs, 1),
        save_path="./models/checkpoints/",
        name_prefix="recurrent_ppo",
    )
    callbacks.append(checkpoint_callback)
    
    # Custom logging callback
    logging_callback = RecurrentRewardCallback(log_freq=5000)
    callbacks.append(logging_callback)
    
    # Train!
    print(f"\n🚀 Starting Recurrent PPO training for {total_timesteps:,} timesteps...")
    print("LSTM-based model may take longer to train but should learn better strategies...\n")
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # Save final model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
    model.save(save_path)
    print(f"\n✅ Recurrent PPO model saved to {save_path}.zip")
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Training Time: {training_time}")
    print(f"Total Games Played: {logging_callback.total_games}")
    print(f"Total Wins: {logging_callback.wins}")
    final_win_rate = (logging_callback.wins / logging_callback.total_games * 100) if logging_callback.total_games > 0 else 0
    print(f"Final Win Rate: {final_win_rate:.1f}%")
    print(f"{'='*70}\n")
    
    # Final evaluation
    print("Running final evaluation (100 games)...")
    evaluate_recurrent_model(model, n_episodes=100)
    
    return model


def evaluate_recurrent_model(model, n_episodes=100):
    """Evaluate the trained recurrent model."""
    env = UnoEnv()
    
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        # Initialize LSTM states
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        while not done and not truncated:
            # Recurrent PPO needs additional parameters
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            episode_start = np.zeros((1,), dtype=bool)
        
        total_reward += episode_reward
        episode_lengths.append(steps)
        
        if info.get("winner") == "player":
            wins += 1
    
    win_rate = wins / n_episodes * 100
    avg_reward = total_reward / n_episodes
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*50}")
    print("📊 Recurrent PPO Evaluation Results")
    print(f"{'='*50}")
    print(f"Episodes: {n_episodes}")
    print(f"Wins: {wins}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"{'='*50}\n")
    
    return {
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_episode_length": avg_length,
    }


def load_and_evaluate(model_path: str, n_episodes: int = 100):
    """Load a saved recurrent model and evaluate it."""
    print(f"Loading model from {model_path}...")
    model = RecurrentPPO.load(model_path)
    print("Model loaded successfully!")
    
    return evaluate_recurrent_model(model, n_episodes)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a Recurrent PPO (LSTM) model for UNO")
    parser.add_argument("--timesteps", "-t", type=int, default=500000,
                        help="Total timesteps to train (default: 500000)")
    parser.add_argument("--envs", "-e", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--save-path", "-s", type=str, default="models/recurrent_ppo_uno",
                        help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--lstm-hidden-size", type=int, default=128,
                        help="LSTM hidden state size (default: 128)")
    parser.add_argument("--lstm-layers", type=int, default=1,
                        help="Number of LSTM layers (default: 1)")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Path to model to evaluate (skip training)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Just evaluate an existing model
        load_and_evaluate(args.evaluate, n_episodes=100)
    else:
        # Train a new model
        model = train_recurrent_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            save_path=args.save_path,
            seed=args.seed,
            lstm_hidden_size=args.lstm_hidden_size,
            n_lstm_layers=args.lstm_layers,
        )
