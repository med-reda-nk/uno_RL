"""
Training script for Stable Baselines3 agents on UNO.
Supports DQN, PPO, and A2C algorithms from pre-built SB3 library.
"""

import os
import sys
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.sb3_agent import SB3UnoAgent, UnoEnv, create_agent, SB3_AVAILABLE
except ImportError as e:
    print(f"Import error: {e}")
    SB3_AVAILABLE = False


def train_sb3_agent(
    algorithm: str = "dqn",
    total_timesteps: int = 100000,
    save_path: str = None,
    eval_freq: int = 10000,
    verbose: bool = True,
):
    """
    Train a Stable Baselines3 agent on UNO.
    
    Parameters:
    -----------
    algorithm : str
        RL algorithm: "dqn", "ppo", or "a2c"
    total_timesteps : int
        Total environment steps to train
    save_path : str
        Path to save the model
    eval_freq : int
        Evaluate every N steps
    verbose : bool
        Print progress
    """
    if not SB3_AVAILABLE:
        print("Error: Stable Baselines3 is not installed.")
        print("Install it with: pip install stable-baselines3[extra] gymnasium")
        return None
    
    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} Agent using Stable Baselines3")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Save Path: {save_path or 'Not saving'}")
    print(f"{'='*60}\n")
    
    # Create agent with algorithm-specific config
    if algorithm == "dqn":
        config = {
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 1000,
            "exploration_fraction": 0.3,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": dict(net_arch=[256, 256, 128]),
            "verbose": 1 if verbose else 0,
        }
    elif algorithm == "ppo":
        config = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            "verbose": 1 if verbose else 0,
        }
    elif algorithm == "a2c":
        config = {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            "verbose": 1 if verbose else 0,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Create and train agent
    agent = create_agent(algorithm=algorithm, config=config)
    
    if save_path is None:
        save_path = f"models/sb3_{algorithm}_uno"
    
    agent.train(
        total_timesteps=total_timesteps,
        save_path=save_path,
        eval_freq=eval_freq,
    )
    
    return agent


def evaluate_sb3_agent(
    model_path: str = None,
    algorithm: str = "dqn",
    n_episodes: int = 100,
):
    """
    Evaluate a trained SB3 agent.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    algorithm : str
        Algorithm used for the model
    n_episodes : int
        Number of episodes to evaluate
    """
    if not SB3_AVAILABLE:
        print("Error: Stable Baselines3 is not installed.")
        return None
    
    if model_path is None:
        model_path = f"models/sb3_{algorithm}_uno"
    
    print(f"\n{'='*60}")
    print(f"Evaluating {algorithm.upper()} Agent")
    print(f"{'='*60}")
    print(f"Model Path: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*60}\n")
    
    # Create agent and load model
    agent = create_agent(algorithm=algorithm)
    agent.load(model_path)
    
    # Evaluate
    metrics = agent.evaluate(n_episodes=n_episodes)
    
    return metrics


def compare_algorithms(timesteps: int = 50000, eval_episodes: int = 100):
    """
    Train and compare different SB3 algorithms.
    
    Parameters:
    -----------
    timesteps : int
        Training timesteps per algorithm
    eval_episodes : int
        Episodes for evaluation
    """
    if not SB3_AVAILABLE:
        print("Error: Stable Baselines3 is not installed.")
        return
    
    algorithms = ["dqn", "ppo", "a2c"]
    results = {}
    
    print(f"\n{'='*60}")
    print("Comparing RL Algorithms on UNO")
    print(f"{'='*60}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Timesteps per algorithm: {timesteps:,}")
    print(f"Evaluation episodes: {eval_episodes}")
    print(f"{'='*60}\n")
    
    for algo in algorithms:
        print(f"\n{'='*40}")
        print(f"Training {algo.upper()}")
        print(f"{'='*40}")
        
        try:
            agent = train_sb3_agent(
                algorithm=algo,
                total_timesteps=timesteps,
                eval_freq=timesteps // 5,
            )
            
            metrics = agent.evaluate(n_episodes=eval_episodes)
            results[algo] = metrics
        except Exception as e:
            print(f"Error training {algo}: {e}")
            results[algo] = {"error": str(e)}
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"{'Algorithm':<15} {'Win Rate':<15} {'Avg Reward':<15}")
    print("-" * 45)
    
    for algo, metrics in results.items():
        if "error" in metrics:
            print(f"{algo.upper():<15} {'Error':<15} {metrics['error']}")
        else:
            win_rate = f"{metrics['win_rate']:.2%}"
            avg_reward = f"{metrics['avg_reward']:.2f}"
            print(f"{algo.upper():<15} {win_rate:<15} {avg_reward:<15}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for UNO using Stable Baselines3")
    parser.add_argument("--algorithm", "-a", type=str, default="dqn",
                        choices=["dqn", "ppo", "a2c"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", "-t", type=int, default=100000,
                        help="Total timesteps to train")
    parser.add_argument("--save-path", "-s", type=str, default=None,
                        help="Path to save the model")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluate every N steps")
    parser.add_argument("--evaluate", "-e", action="store_true",
                        help="Only evaluate (don't train)")
    parser.add_argument("--model-path", "-m", type=str, default=None,
                        help="Path to model for evaluation")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of episodes for evaluation")
    parser.add_argument("--compare", "-c", action="store_true",
                        help="Compare all algorithms")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms(timesteps=args.timesteps, eval_episodes=args.eval_episodes)
    elif args.evaluate:
        evaluate_sb3_agent(
            model_path=args.model_path,
            algorithm=args.algorithm,
            n_episodes=args.eval_episodes,
        )
    else:
        agent = train_sb3_agent(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            eval_freq=args.eval_freq,
        )
        
        if agent:
            print("\nRunning final evaluation...")
            agent.evaluate(n_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
