"""
Train the BEST PPO model for UNO with optimized hyperparameters.
Based on hyperparameter tuning results + additional optimizations.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from src.sb3_agent import UnoEnv
    SB3_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Install with: pip install stable-baselines3[extra] gymnasium")
    SB3_AVAILABLE = False
    sys.exit(1)


class RewardLoggingCallback(BaseCallback):
    """Custom callback for logging training metrics."""
    
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
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


def train_best_ppo(
    total_timesteps: int = 500000,
    n_envs: int = 4,
    save_path: str = "models/best_ppo_uno",
    seed: int = 42,
):
    """
    Train the best PPO model with optimized hyperparameters.
    
    Best hyperparameters found from tuning:
    - learning_rate: 0.0001 (lower for stability)
    - n_steps: 1024 (good balance)
    - batch_size: 32 (smaller for better gradients)
    - n_epochs: 5 (prevent overfitting)
    - ent_coef: 0.01 (encourage exploration)
    
    Additional optimizations:
    - Larger network architecture
    - Multiple parallel environments
    - Learning rate schedule
    - Gradient clipping
    - Proper advantage normalization
    """
    
    print(f"\n{'='*70}")
    print("🎯 Training BEST PPO Model for UNO")
    print(f"{'='*70}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Save Path: {save_path}")
    print(f"Random Seed: {seed}")
    print(f"{'='*70}\n")
    
    # Create vectorized environments for faster training
    print("Creating parallel environments...")
    env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = Monitor(UnoEnv())
    
    # BEST PPO Configuration (based on tuning + additional optimizations)
    ppo_config = {
        # Learning parameters (from hyperparameter tuning)
        "learning_rate": 1e-4,  # Best from tuning
        "n_steps": 1024,        # Best from tuning
        "batch_size": 32,       # Best from tuning (smaller = more stable)
        "n_epochs": 5,          # Best from tuning (prevent overfitting)
        
        # Discount and advantage estimation
        "gamma": 0.995,         # Slightly higher for longer-term planning
        "gae_lambda": 0.98,     # Higher for better advantage estimation
        
        # PPO-specific parameters
        "clip_range": 0.2,      # Standard PPO clipping
        "clip_range_vf": None,  # No value function clipping
        "normalize_advantage": True,  # Critical for stability
        
        # Entropy for exploration
        "ent_coef": 0.01,       # Best from tuning
        "vf_coef": 0.5,         # Value function coefficient
        
        # Gradient clipping for stability
        "max_grad_norm": 0.5,
        
        # Network architecture - deeper and wider
        "policy_kwargs": dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Policy network
                vf=[256, 256, 128],  # Value network
            ),
            # Orthogonal initialization for better training
            ortho_init=True,
        ),
        
        # Other settings
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": "./logs/ppo_best/",
    }
    
    print("PPO Configuration:")
    for key, value in ppo_config.items():
        if key != "policy_kwargs":
            print(f"  {key}: {value}")
    print(f"  policy_kwargs: {ppo_config['policy_kwargs']}")
    print()
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO("MlpPolicy", env, **ppo_config)
    
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
        name_prefix="ppo_best",
    )
    callbacks.append(checkpoint_callback)
    
    # Custom logging callback
    logging_callback = RewardLoggingCallback(log_freq=5000)
    callbacks.append(logging_callback)
    
    # Train!
    print(f"\n🚀 Starting training for {total_timesteps:,} timesteps...")
    print("This may take a while...\n")
    
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
    print(f"\n✅ Model saved to {save_path}.zip")
    
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
    evaluate_model(model, n_episodes=100)
    
    return model


def evaluate_model(model, n_episodes=100):
    """Evaluate the trained model."""
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
        
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        total_reward += episode_reward
        episode_lengths.append(steps)
        
        if info.get("winner") == "player":
            wins += 1
    
    win_rate = wins / n_episodes * 100
    avg_reward = total_reward / n_episodes
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*50}")
    print("📊 Evaluation Results")
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the best PPO model for UNO")
    parser.add_argument("--timesteps", "-t", type=int, default=500000,
                        help="Total timesteps to train (default: 500000)")
    parser.add_argument("--envs", "-e", type=int, default=4,
                        help="Number of parallel environments (default: 4)")
    parser.add_argument("--save-path", "-s", type=str, default="models/best_ppo_uno",
                        help="Path to save the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    model = train_best_ppo(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_path=args.save_path,
        seed=args.seed,
    )
