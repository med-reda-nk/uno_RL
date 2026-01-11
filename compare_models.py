"""
Model Comparison Script for UNO RL Agents
Compare different RL algorithms and configurations to find the best performing agent.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "training"))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False
from src.sb3_agent import UnoEnv

# Try to import Enhanced UNO environment
try:
    from training.train_enhanced_rppo import EnhancedUnoEnv
    HAS_ENHANCED_ENV = True
except ImportError:
    try:
        from train_enhanced_rppo import EnhancedUnoEnv
        HAS_ENHANCED_ENV = True
    except ImportError:
        HAS_ENHANCED_ENV = False


class ModelComparator:
    """Compare different RL models for UNO game."""
    
    def __init__(self, models_dir: str = "models", results_dir: str = "comparison_results"):
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        self.results: List[Dict] = []
        self.env = UnoEnv()
        self.enhanced_env = EnhancedUnoEnv(opponent_strength=0.2) if HAS_ENHANCED_ENV else None
    
    def load_model(self, model_path: str, algorithm: str = "ppo"):
        """Load a trained model."""
        algo_map = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
        if HAS_RECURRENT_PPO:
            algo_map["recurrentppo"] = RecurrentPPO
        algo_class = algo_map.get(algorithm.lower(), PPO)
        return algo_class.load(model_path)
    
    def evaluate_model(self, model, n_episodes: int = 100, 
                       deterministic: bool = True) -> Dict:
        """Evaluate a model and return detailed statistics."""
        wins = 0
        losses = 0
        total_rewards = []
        episode_lengths = []
        cards_remaining_wins = []
        cards_remaining_losses = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Check if agent won (player hand is empty)
            if len(self.env.player_hand) == 0:
                wins += 1
                cards_remaining_wins.append(len(self.env.opponent_hand))
            else:
                losses += 1
                cards_remaining_losses.append(len(self.env.player_hand))
        
        return {
            "wins": wins,
            "losses": losses,
            "win_rate": wins / n_episodes * 100,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "avg_cards_remaining_win": np.mean(cards_remaining_wins) if cards_remaining_wins else 0,
            "avg_cards_remaining_loss": np.mean(cards_remaining_losses) if cards_remaining_losses else 0,
        }
    
    def evaluate_recurrent_model(self, model, n_episodes: int = 100,
                                  deterministic: bool = True) -> Dict:
        """Evaluate a recurrent (LSTM) model with proper state handling."""
        wins = 0
        losses = 0
        total_rewards = []
        episode_lengths = []
        cards_remaining_wins = []
        cards_remaining_losses = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            # Initialize LSTM states
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            
            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=deterministic
                )
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                episode_start = np.zeros((1,), dtype=bool)
            
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
            
            # Check winner from info or hand size
            if info.get("winner") == "player" or len(self.env.player_hand) == 0:
                wins += 1
                cards_remaining_wins.append(len(self.env.opponent_hand))
            else:
                losses += 1
                cards_remaining_losses.append(len(self.env.player_hand))
        
        return {
            "wins": wins,
            "losses": losses,
            "win_rate": wins / n_episodes * 100,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "avg_cards_remaining_win": np.mean(cards_remaining_wins) if cards_remaining_wins else 0,
            "avg_cards_remaining_loss": np.mean(cards_remaining_losses) if cards_remaining_losses else 0,
        }
    
    def evaluate_enhanced_model(self, model, n_episodes: int = 100,
                                 deterministic: bool = True, opponent_strength: float = 0.2) -> Dict:
        """Evaluate Enhanced RPPO model with its specific environment."""
        if not HAS_ENHANCED_ENV:
            raise ImportError("EnhancedUnoEnv not available")
        
        env = EnhancedUnoEnv(opponent_strength=opponent_strength)
        
        wins = 0
        losses = 0
        total_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            
            while not done:
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=deterministic
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
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
    
    def compare_existing_models(self, n_episodes: int = 100) -> pd.DataFrame:
        """Compare all existing models in the models directory."""
        print("\n" + "="*60)
        print("COMPARING ALL EXISTING MODELS")
        print("="*60)
        
        # All available models with their algorithms
        model_files = {
            # Self-Play Champion (NEW - target 70%+)
            "Self-Play Champion": ("selfplay_champion.zip", "recurrentppo"),
            
            # Standard SB3 models
            "PPO": ("sb3_ppo_uno.zip", "ppo"),
            "DQN": ("sb3_dqn_uno.zip", "dqn"),
            "A2C": ("sb3_a2c_uno.zip", "a2c"),
            "PPO Best Config": ("sb3_ppo_best_config.zip", "ppo"),
            
            # Recurrent PPO models
            "Recurrent PPO": ("recurrent_ppo_uno.zip", "recurrentppo"),
            "SB3 Recurrent PPO": ("sb3_recurrentppo_uno.zip", "recurrentppo"),
            "Best Recurrent PPO": ("best_recurrent_ppo_uno.zip", "recurrentppo"),
            
            # Optimal and Enhanced models
            "Optimal Recurrent PPO": ("optimal_recurrent_ppo.zip", "recurrentppo"),
            "Best Optimal RPPO": ("best_optimal_rppo/best_model.zip", "recurrentppo"),
            "Enhanced RPPO": ("enhanced_rppo.zip", "enhanced_rppo"),
            "Best Enhanced RPPO": ("best_enhanced_rppo/best_model.zip", "enhanced_rppo"),
            
            # Other best models
            "Best PPO": ("best_ppo_uno.zip", "ppo"),
            "Best Model": ("best_model.zip", "ppo"),
        }
        
        results = []
        
        for name, (filename, algo) in model_files.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                print(f"\n📊 Evaluating {name}...")
                
                try:
                    model = self.load_model(path, algo if algo != "enhanced_rppo" else "recurrentppo")
                    
                    # Use special evaluation for enhanced and recurrent models
                    if algo == "enhanced_rppo":
                        stats = self.evaluate_enhanced_model(model, n_episodes)
                    elif algo == "recurrentppo":
                        stats = self.evaluate_recurrent_model(model, n_episodes)
                    else:
                        stats = self.evaluate_model(model, n_episodes)
                    
                    stats["model_name"] = name
                    stats["algorithm"] = algo.upper()
                    stats["model_path"] = path
                    results.append(stats)
                    
                    print(f"   Win Rate: {stats['win_rate']:.1f}%")
                    print(f"   Avg Reward: {stats['avg_reward']:.2f}")
                except Exception as e:
                    print(f"   ❌ Failed to load: {e}")
            else:
                print(f"\n⚠️  {name} not found at {path}")
        
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("win_rate", ascending=False)
            return df
        return pd.DataFrame()
    
    def train_and_compare(self, algorithms: List[str] = ["ppo", "dqn", "a2c", "recurrentppo"],
                          timesteps: int = 100000, n_eval_episodes: int = 100) -> pd.DataFrame:
        """Train multiple algorithms and compare their performance."""
        print("\n" + "="*60)
        print("TRAINING AND COMPARING ALGORITHMS")
        print("="*60)
        
        results = []
        
        for algo_name in algorithms:
            print(f"\n🏋️ Training {algo_name.upper()}...")
            
            env = UnoEnv()
            
            if algo_name.lower() == "ppo":
                model = PPO(
                    "MlpPolicy", env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    verbose=0
                )
            elif algo_name.lower() == "dqn":
                model = DQN(
                    "MlpPolicy", env,
                    learning_rate=1e-4,
                    buffer_size=100000,
                    learning_starts=1000,
                    batch_size=64,
                    gamma=0.99,
                    exploration_fraction=0.2,
                    exploration_final_eps=0.05,
                    verbose=0
                )
            elif algo_name.lower() == "a2c":
                model = A2C(
                    "MlpPolicy", env,
                    learning_rate=7e-4,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    ent_coef=0.01,
                    verbose=0
                )
            elif algo_name.lower() == "recurrentppo":
                if not HAS_RECURRENT_PPO:
                    print(f"   ⚠️  sb3_contrib not installed, skipping RecurrentPPO")
                    continue
                model = RecurrentPPO(
                    "MlpLstmPolicy", env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    verbose=0
                )
            else:
                continue
            
            # Train
            model.learn(total_timesteps=timesteps, progress_bar=True)
            
            # Save
            save_path = os.path.join(self.models_dir, f"sb3_{algo_name}_uno.zip")
            model.save(save_path)
            
            # Evaluate
            print(f"📊 Evaluating {algo_name.upper()}...")
            stats = self.evaluate_model(model, n_eval_episodes)
            stats["model_name"] = algo_name.upper()
            stats["algorithm"] = algo_name.upper()
            stats["timesteps"] = timesteps
            stats["model_path"] = save_path
            results.append(stats)
            
            print(f"   Win Rate: {stats['win_rate']:.1f}%")
            print(f"   Avg Reward: {stats['avg_reward']:.2f}")
            
            env.close()
        
        df = pd.DataFrame(results)
        df = df.sort_values("win_rate", ascending=False)
        return df
    
    def hyperparameter_search(self, algorithm: str = "ppo", 
                              n_trials: int = 5, timesteps: int = 50000,
                              n_eval_episodes: int = 100) -> pd.DataFrame:
        """Search for best hyperparameters for a given algorithm."""
        print("\n" + "="*60)
        print(f"HYPERPARAMETER SEARCH FOR {algorithm.upper()}")
        print("="*60)
        
        results = []
        
        # Define hyperparameter configurations
        if algorithm.lower() == "ppo":
            configs = [
                {"learning_rate": 1e-4, "n_steps": 1024, "batch_size": 32, "n_epochs": 5, "ent_coef": 0.01},
                {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 10, "ent_coef": 0.01},
                {"learning_rate": 5e-4, "n_steps": 2048, "batch_size": 128, "n_epochs": 10, "ent_coef": 0.02},
                {"learning_rate": 1e-4, "n_steps": 4096, "batch_size": 64, "n_epochs": 20, "ent_coef": 0.005},
                {"learning_rate": 2e-4, "n_steps": 2048, "batch_size": 64, "n_epochs": 15, "ent_coef": 0.01},
            ]
        elif algorithm.lower() == "dqn":
            configs = [
                {"learning_rate": 1e-4, "buffer_size": 50000, "batch_size": 32, "exploration_fraction": 0.2},
                {"learning_rate": 5e-5, "buffer_size": 100000, "batch_size": 64, "exploration_fraction": 0.3},
                {"learning_rate": 1e-4, "buffer_size": 100000, "batch_size": 128, "exploration_fraction": 0.15},
                {"learning_rate": 2e-4, "buffer_size": 200000, "batch_size": 64, "exploration_fraction": 0.25},
                {"learning_rate": 5e-5, "buffer_size": 150000, "batch_size": 32, "exploration_fraction": 0.2},
            ]
        else:
            print(f"Hyperparameter search not implemented for {algorithm}")
            return pd.DataFrame()
        
        for i, config in enumerate(configs[:n_trials]):
            print(f"\n🔬 Trial {i+1}/{min(n_trials, len(configs))}: {config}")
            
            env = UnoEnv()
            
            if algorithm.lower() == "ppo":
                model = PPO("MlpPolicy", env, verbose=0, **config)
            elif algorithm.lower() == "dqn":
                model = DQN("MlpPolicy", env, verbose=0, **config)
            
            model.learn(total_timesteps=timesteps, progress_bar=True)
            
            stats = self.evaluate_model(model, n_eval_episodes)
            stats["config"] = str(config)
            stats["trial"] = i + 1
            results.append(stats)
            
            print(f"   Win Rate: {stats['win_rate']:.1f}%")
            
            # Save best model
            if not results or stats["win_rate"] >= max(r["win_rate"] for r in results):
                best_path = os.path.join(self.models_dir, f"sb3_{algorithm}_best_config.zip")
                model.save(best_path)
                print(f"   💾 New best model saved!")
            
            env.close()
        
        df = pd.DataFrame(results)
        df = df.sort_values("win_rate", ascending=False)
        return df
    
    def generate_report(self, results_df: pd.DataFrame, report_name: str = "comparison"):
        """Generate a comparison report with visualizations."""
        if results_df.empty:
            print("No results to report.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = os.path.join(self.results_dir, f"{report_name}_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n📄 Results saved to: {csv_path}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("UNO RL Model Comparison", fontsize=14, fontweight='bold')
        
        # Win Rate Bar Chart
        ax1 = axes[0, 0]
        model_names = results_df["model_name"].tolist() if "model_name" in results_df else [f"Model {i}" for i in range(len(results_df))]
        win_rates = results_df["win_rate"].tolist()
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        bars = ax1.bar(model_names, win_rates, color=colors)
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_title("Win Rate Comparison")
        ax1.set_ylim(0, 100)
        for bar, rate in zip(bars, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Average Reward Bar Chart
        ax2 = axes[0, 1]
        avg_rewards = results_df["avg_reward"].tolist()
        bars = ax2.bar(model_names, avg_rewards, color=colors)
        ax2.set_ylabel("Average Reward")
        ax2.set_title("Average Reward Comparison")
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Episode Length
        ax3 = axes[1, 0]
        if "avg_episode_length" in results_df:
            ep_lengths = results_df["avg_episode_length"].tolist()
            ax3.bar(model_names, ep_lengths, color=colors)
            ax3.set_ylabel("Average Episode Length")
            ax3.set_title("Episode Length Comparison")
        
        # Win/Loss breakdown
        ax4 = axes[1, 1]
        wins = results_df["wins"].tolist()
        losses = results_df["losses"].tolist()
        x = np.arange(len(model_names))
        width = 0.35
        ax4.bar(x - width/2, wins, width, label='Wins', color='#2ecc71')
        ax4.bar(x + width/2, losses, width, label='Losses', color='#e74c3c')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.set_ylabel("Count")
        ax4.set_title("Wins vs Losses")
        ax4.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, f"{report_name}_{timestamp}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {fig_path}")
        plt.close()
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(results_df[["model_name", "win_rate", "avg_reward", "wins", "losses"]].to_string(index=False))
        
        best = results_df.iloc[0]
        print(f"\n🏆 Best Model: {best.get('model_name', 'Unknown')} with {best['win_rate']:.1f}% win rate")


def main():
    parser = argparse.ArgumentParser(description="Compare UNO RL Models")
    parser.add_argument("--mode", type=str, default="compare", 
                       choices=["compare", "train", "search", "all"],
                       help="Comparison mode")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of evaluation episodes")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Training timesteps for train mode")
    parser.add_argument("--trials", type=int, default=5,
                       help="Number of trials for hyperparameter search")
    parser.add_argument("--algorithm", type=str, default="ppo",
                       help="Algorithm for hyperparameter search")
    
    args = parser.parse_args()
    
    comparator = ModelComparator()
    
    if args.mode == "compare":
        results = comparator.compare_existing_models(args.episodes)
        if not results.empty:
            comparator.generate_report(results, "existing_models")
    
    elif args.mode == "train":
        results = comparator.train_and_compare(
            algorithms=["ppo", "dqn", "a2c", "recurrentppo"],
            timesteps=args.timesteps,
            n_eval_episodes=args.episodes
        )
        comparator.generate_report(results, "trained_models")
    
    elif args.mode == "search":
        results = comparator.hyperparameter_search(
            algorithm=args.algorithm,
            n_trials=args.trials,
            timesteps=args.timesteps,
            n_eval_episodes=args.episodes
        )
        comparator.generate_report(results, f"hyperparam_{args.algorithm}")
    
    elif args.mode == "all":
        # Compare existing
        print("\n" + "🔍 "*20)
        print("PHASE 1: Comparing Existing Models")
        results1 = comparator.compare_existing_models(args.episodes)
        if not results1.empty:
            comparator.generate_report(results1, "existing_models")
        
        # Train new
        print("\n" + "🔍 "*20)
        print("PHASE 2: Training New Models")
        results2 = comparator.train_and_compare(
            algorithms=["ppo", "dqn", "a2c", "recurrentppo"],
            timesteps=args.timesteps,
            n_eval_episodes=args.episodes
        )
        comparator.generate_report(results2, "trained_models")
        
        # Hyperparameter search for best
        print("\n" + "🔍 "*20)
        print("PHASE 3: Hyperparameter Search")
        results3 = comparator.hyperparameter_search(
            algorithm="ppo",
            n_trials=args.trials,
            timesteps=args.timesteps,
            n_eval_episodes=args.episodes
        )
        comparator.generate_report(results3, "hyperparam_ppo")


if __name__ == "__main__":
    main()
