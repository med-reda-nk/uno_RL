"""
Train a "Best" Recurrent PPO model for UNO with optimized hyperparameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from typing import Callable
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from src.sb3_agent import UnoEnv
    RPPO_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
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


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule function.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func


def make_env(rank, seed=0):
    """Create a wrapped, monitored UnoEnv."""
    def _init():
        env = UnoEnv()
        env = Monitor(env)
        return env
    set_random_seed(seed + rank)
    return _init


def evaluate_recurrent_model_stats(model, n_episodes=100) -> dict:
    """Evaluate constraints and return full stats dict for CSV."""
    env = UnoEnv()
    
    wins = 0
    losses = 0
    total_rewards = []
    episode_lengths = []
    
    print(f"Evaluating model over {n_episodes} episodes...")
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
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if info.get("winner") == "player":
            wins += 1
        else:
            losses += 1
            
    return {
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n_episodes * 100,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_episode_length": np.mean(episode_lengths),
    }


def train_best_recurrent_ppo(
    total_timesteps: int = 500000,  # INCREASED to 500k
    n_envs: int = 8,
    save_path: str = "models/best_recurrent_ppo_uno",
    seed: int = 42,
    lstm_hidden_size: int = 256,
    n_lstm_layers: int = 2,
):
    print(f"\n{'='*70}")
    print("🧠 Training BEST Recurrent PPO (LSTM) Model for UNO")
    print(f"{'='*70}")
    
    # Create vectorized environments
    print(f"Creating {n_envs} parallel environments...")
    env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])
    
    # Normalize rewards (crucial for PPO)
    # Re-enabled based on high value loss analysis
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.995)
    
    # Evaluation environment
    eval_env = DummyVecEnv([make_env(i, seed + 100) for i in range(1)])
    
    # OPIMIZED Configuration (Attempt 2: Shared LSTM + Normalization)
    rppo_config = {
        "learning_rate": linear_schedule(3e-4), # Back to standard PPO rate
        "n_steps": 1024,           # Reduced to match PPO baseline
        "batch_size": 64,          # Reduced to match PPO baseline
        "n_epochs": 10,            # Back to 10
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": dict(
            lstm_hidden_size=128,   # REDUCED from 256 for better data efficiency
            n_lstm_layers=2,
            shared_lstm=True,       # ENABLED shared LSTM
            enable_critic_lstm=False, # MUST be False if shared_lstm is True
            net_arch=dict(
                pi=[128, 128],      # REDUCED to match hidden size
                vf=[128, 128],
            ),
        ),
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": "./logs/best_recurrent_ppo/",
    }
    
    print("Configuration:")
    for k, v in rppo_config.items():
        if k != "policy_kwargs":
            print(f"  {k}: {v}")
    
    print("Initializing model...")
    model = RecurrentPPO("MlpLstmPolicy", env, **rppo_config)
    
    # Callbacks
    callbacks = [
        RecurrentRewardCallback(log_freq=5000),
        CheckpointCallback(save_freq=50000 // n_envs, save_path="./models/checkpoints/", name_prefix="best_rppo"),
    ]
    
    print(f"\n🚀 Starting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    
    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl") # Save normalization stats
    print(f"\n✅ Model saved to {save_path}.zip")
    print(f"✅ Normalization stats saved to {save_path}_vecnormalize.pkl")
    
    # Generate Analysis Data
    print("\n📊 Generating Analysis Data...")
    stats = evaluate_recurrent_model_stats(model, n_episodes=200)
    
    stats["model_name"] = "Best Recurrent PPO"
    stats["algorithm"] = "RECURRENTPPO"
    stats["model_path"] = f"{save_path}.zip"
    stats["timesteps"] = total_timesteps
    stats["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save results to CSV for analysis.ipynb
    results_dir = "comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "best_recurrent_results.csv")
    
    df = pd.DataFrame([stats])
    df.to_csv(csv_path, index=False)
    print(f"📄 Analysis data saved to {csv_path}")
    
    print(f"\nFinal Stats: Win Rate = {stats['win_rate']:.1f}%")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--envs", type=int, default=8)
    args = parser.parse_args()
    
    train_best_recurrent_ppo(total_timesteps=args.timesteps, n_envs=args.envs)
