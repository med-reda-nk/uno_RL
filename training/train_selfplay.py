"""
Self-Play Training with Curriculum Learning
============================================
Advanced training script to improve win rate beyond 60%

Features:
1. Self-Play Training - Train against copies of itself
2. Curriculum Learning - Gradually increase opponent difficulty
3. Multiplayer Training - Train in 3-4 player environments
4. Enhanced Reward Shaping - Better reward signals
5. Population-Based Training - Multiple agents competing

Target: 70%+ win rate
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Error: stable-baselines3 required. Install: pip install stable-baselines3")
    sys.exit(1)

try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_AVAILABLE = True
except ImportError:
    RECURRENT_AVAILABLE = False
    print("Warning: sb3-contrib not available. Install: pip install sb3-contrib")

from src.multiplayer_env import MultiplayerUnoEnv, ThreePlayerUnoEnv, FourPlayerUnoEnv
from src.sb3_agent import UnoEnv


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play training.
    Periodically updates opponent models with current policy.
    """
    
    def __init__(
        self, 
        update_freq: int = 10000,
        save_path: str = "models/selfplay",
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_path = save_path
        self.opponent_versions = []
        self.best_win_rate = 0.0
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Update opponents periodically
        if self.n_calls % self.update_freq == 0:
            # Save current policy as new opponent
            version_path = os.path.join(self.save_path, f"opponent_v{len(self.opponent_versions)}.zip")
            self.model.save(version_path)
            self.opponent_versions.append(version_path)
            
            if self.verbose:
                print(f"\n[SelfPlay] Saved opponent version {len(self.opponent_versions)} at step {self.n_calls}")
            
            # Update environment with new opponent pool
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    if hasattr(env, 'env') and hasattr(env.env, 'opponent_models'):
                        # Load latest opponents into environment
                        self._update_env_opponents(env.env)
        
        return True
    
    def _update_env_opponents(self, env):
        """Update environment with latest opponent models."""
        if len(self.opponent_versions) > 0:
            # Use recent versions as opponents
            recent_versions = self.opponent_versions[-3:]  # Last 3 versions
            
            opponent_models = []
            for path in recent_versions:
                try:
                    model = PPO.load(path)
                    opponent_models.append(model)
                except:
                    pass
            
            env.opponent_models = opponent_models
            env.curriculum_level = min(0.3 + len(self.opponent_versions) * 0.05, 0.9)


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.
    Gradually increases opponent difficulty based on win rate.
    """
    
    def __init__(
        self,
        eval_freq: int = 5000,
        target_win_rate: float = 0.55,
        curriculum_step: float = 0.1,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.target_win_rate = target_win_rate
        self.curriculum_step = curriculum_step
        self.current_level = 0.0
        self.recent_wins = []
        self.window_size = 100
    
    def _on_step(self) -> bool:
        # Track wins
        if 'winner' in self.locals.get('infos', [{}])[0]:
            won = self.locals['infos'][0].get('winner') == 'agent'
            self.recent_wins.append(1 if won else 0)
            if len(self.recent_wins) > self.window_size:
                self.recent_wins.pop(0)
        
        # Update curriculum periodically
        if self.n_calls % self.eval_freq == 0 and len(self.recent_wins) >= 50:
            win_rate = sum(self.recent_wins) / len(self.recent_wins)
            
            if win_rate > self.target_win_rate:
                # Increase difficulty
                self.current_level = min(self.current_level + self.curriculum_step, 1.0)
                if self.verbose:
                    print(f"\n[Curriculum] Win rate {win_rate:.2%} > {self.target_win_rate:.2%}")
                    print(f"[Curriculum] Increasing difficulty to {self.current_level:.2f}")
            
            # Update environments
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    if hasattr(env, 'env') and hasattr(env.env, 'curriculum_level'):
                        env.env.curriculum_level = self.current_level
        
        return True


class WinRateCallback(BaseCallback):
    """Track and log win rate during training."""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.games = 0
        self.win_history = []
    
    def _on_step(self) -> bool:
        # Track game outcomes
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if 'winner' in info:
                self.games += 1
                if info['winner'] == 'agent':
                    self.wins += 1
        
        # Log periodically
        if self.n_calls % self.log_freq == 0 and self.games > 0:
            win_rate = self.wins / self.games
            self.win_history.append(win_rate)
            
            if self.verbose:
                print(f"\n[Stats] Step {self.n_calls}: Win Rate = {win_rate:.2%} ({self.wins}/{self.games})")
            
            # Reset counters
            self.wins = 0
            self.games = 0
        
        return True


def create_env(
    num_players: int = 2,
    opponent_models: Optional[List] = None,
    curriculum_level: float = 0.0
) -> gym.Env:
    """Create and wrap the UNO environment."""
    
    if num_players == 2:
        env = UnoEnv()
    else:
        env = MultiplayerUnoEnv(
            num_players=num_players,
            opponent_models=opponent_models,
            curriculum_level=curriculum_level
        )
    
    return Monitor(env)


def train_selfplay(
    total_timesteps: int = 500_000,
    num_players: int = 2,
    use_recurrent: bool = True,
    save_path: str = "models/selfplay_champion",
    log_dir: str = "logs/selfplay",
    n_envs: int = 4,
    seed: int = 42,
):
    """
    Train using self-play for improved performance.
    
    Args:
        total_timesteps: Total training steps
        num_players: Number of players (2-4)
        use_recurrent: Use LSTM-based RecurrentPPO
        save_path: Path to save final model
        log_dir: Directory for logs
        n_envs: Number of parallel environments
        seed: Random seed
    """
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SELF-PLAY TRAINING FOR UNO RL                      ║
╠══════════════════════════════════════════════════════════════╣
║  Target: 70%+ Win Rate                                       ║
║  Method: Self-Play + Curriculum Learning                     ║
║  Players: {num_players}                                                  ║
║  Algorithm: {'RecurrentPPO (LSTM)' if use_recurrent else 'PPO'}                                  ║
║  Timesteps: {total_timesteps:,}                                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
    
    # Create vectorized environment
    def make_env():
        return create_env(num_players=num_players, curriculum_level=0.0)
    
    if n_envs > 1:
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env])
    
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Model configuration optimized for self-play
    if use_recurrent and RECURRENT_AVAILABLE:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,  # Higher gamma for long-term planning
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,  # Higher entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                lstm_hidden_size=256,  # Larger LSTM
                n_lstm_layers=2,       # Deeper LSTM
                net_arch=dict(pi=[256, 128], vf=[256, 128]),
                enable_critic_lstm=True,
                shared_lstm=False,
            ),
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
        )
        model_type = "RecurrentPPO"
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
            ),
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
        )
        model_type = "PPO"
    
    # Callbacks
    callbacks = [
        WinRateCallback(log_freq=5000, verbose=1),
        CurriculumCallback(eval_freq=10000, target_win_rate=0.55, verbose=1),
        SelfPlayCallback(update_freq=25000, save_path=os.path.join(log_dir, "opponents"), verbose=1),
        CheckpointCallback(
            save_freq=50000,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="selfplay_model"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best"),
            log_path=log_dir,
            eval_freq=25000,
            n_eval_episodes=100,
            deterministic=True,
        ),
    ]
    
    # Train
    print(f"\nStarting {model_type} self-play training...")
    print(f"Monitor with: tensorboard --logdir {log_dir}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # Save final model
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_env_final = DummyVecEnv([make_env])
    mean_reward, std_reward = evaluate_policy(model, eval_env_final, n_eval_episodes=200)
    print(f"Final Performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model


def train_population(
    population_size: int = 5,
    generations: int = 10,
    timesteps_per_gen: int = 100_000,
    num_players: int = 4,
    save_dir: str = "models/population",
):
    """
    Population-based training for multiplayer UNO.
    Multiple agents evolve together through competition.
    """
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        POPULATION-BASED TRAINING FOR UNO RL                  ║
╠══════════════════════════════════════════════════════════════╣
║  Population Size: {population_size}                                           ║
║  Generations: {generations}                                               ║
║  Players per Game: {num_players}                                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize population
    population = []
    for i in range(population_size):
        env = DummyVecEnv([lambda: create_env(num_players=2)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            verbose=0,
            seed=42 + i,
        )
        population.append({"model": model, "env": env, "wins": 0, "games": 0})
        print(f"Initialized agent {i+1}/{population_size}")
    
    # Evolution loop
    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/{generations}")
        print(f"{'='*60}")
        
        # Train each agent
        for i, agent in enumerate(population):
            print(f"\nTraining agent {i+1}...")
            agent["model"].learn(total_timesteps=timesteps_per_gen, progress_bar=True)
        
        # Tournament to evaluate fitness
        print("\nRunning tournament...")
        for agent in population:
            agent["wins"] = 0
            agent["games"] = 0
        
        # Round-robin tournament
        for i in range(population_size):
            for j in range(i + 1, population_size):
                # Battle agents i vs j
                wins_i, wins_j = battle_agents(
                    population[i]["model"], 
                    population[j]["model"],
                    num_games=50
                )
                population[i]["wins"] += wins_i
                population[i]["games"] += wins_i + wins_j
                population[j]["wins"] += wins_j
                population[j]["games"] += wins_i + wins_j
        
        # Rank by win rate
        for agent in population:
            agent["win_rate"] = agent["wins"] / max(agent["games"], 1)
        
        population.sort(key=lambda x: x["win_rate"], reverse=True)
        
        # Print leaderboard
        print("\nLeaderboard:")
        for i, agent in enumerate(population):
            print(f"  {i+1}. Win Rate: {agent['win_rate']:.2%} ({agent['wins']}/{agent['games']})")
        
        # Save best model
        best_model = population[0]["model"]
        best_model.save(os.path.join(save_dir, f"gen{gen+1}_champion.zip"))
        
        # Replace worst with mutated best (simple evolution)
        if len(population) > 2:
            # Keep top 60%, replace bottom 40% with mutated copies of top
            keep_count = int(population_size * 0.6)
            for i in range(keep_count, population_size):
                # Clone from top performers
                source_idx = i % keep_count
                source_model = population[source_idx]["model"]
                
                # Create new environment
                new_env = DummyVecEnv([lambda: create_env(num_players=2)])
                new_env = VecNormalize(new_env, norm_obs=True, norm_reward=True)
                
                # Create new model with slightly different hyperparameters
                lr_mutate = 3e-4 * (0.8 + 0.4 * np.random.random())
                ent_mutate = 0.01 * (0.5 + np.random.random())
                
                new_model = PPO(
                    "MlpPolicy",
                    new_env,
                    learning_rate=lr_mutate,
                    ent_coef=ent_mutate,
                    n_steps=1024,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.99,
                    verbose=0,
                )
                
                population[i] = {"model": new_model, "env": new_env, "wins": 0, "games": 0}
    
    # Save final champion
    final_champion = population[0]["model"]
    final_champion.save(os.path.join(save_dir, "final_champion.zip"))
    
    print(f"\n{'='*60}")
    print(f"Population training complete!")
    print(f"Champion saved to: {os.path.join(save_dir, 'final_champion.zip')}")
    print(f"{'='*60}")
    
    return final_champion


def battle_agents(model1, model2, num_games: int = 100) -> tuple:
    """
    Battle two trained models against each other.
    
    Returns:
        (wins_model1, wins_model2)
    """
    env = UnoEnv()
    wins1, wins2 = 0, 0
    
    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        
        # Alternate who goes first
        model1_turn = (game % 2 == 0)
        
        while not done:
            if model1_turn:
                action, _ = model1.predict(obs, deterministic=True)
            else:
                action, _ = model2.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            model1_turn = not model1_turn
        
        # Determine winner based on final reward
        if reward > 0:
            if (game % 2 == 0) == model1_turn:
                wins2 += 1
            else:
                wins1 += 1
        else:
            if (game % 2 == 0) == model1_turn:
                wins1 += 1
            else:
                wins2 += 1
    
    return wins1, wins2


def main():
    parser = argparse.ArgumentParser(description="Self-Play Training for UNO RL")
    parser.add_argument("--mode", type=str, default="selfplay", 
                       choices=["selfplay", "population", "curriculum"],
                       help="Training mode")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4],
                       help="Number of players")
    parser.add_argument("--recurrent", action="store_true", default=True,
                       help="Use RecurrentPPO with LSTM")
    parser.add_argument("--no-recurrent", dest="recurrent", action="store_false",
                       help="Use standard PPO")
    parser.add_argument("--save-path", type=str, default="models/selfplay_champion",
                       help="Path to save trained model")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    if args.mode == "selfplay":
        train_selfplay(
            total_timesteps=args.timesteps,
            num_players=args.players,
            use_recurrent=args.recurrent,
            save_path=args.save_path,
            seed=args.seed,
        )
    elif args.mode == "population":
        train_population(
            population_size=5,
            generations=10,
            timesteps_per_gen=args.timesteps // 10,
            num_players=args.players,
        )
    elif args.mode == "curriculum":
        # Curriculum learning with increasing difficulty
        train_selfplay(
            total_timesteps=args.timesteps,
            num_players=args.players,
            use_recurrent=args.recurrent,
            save_path=args.save_path.replace(".zip", "_curriculum.zip"),
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
