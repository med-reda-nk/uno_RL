"""
UNO RL - Main Runner Script
Supports both legacy Q-learning and modern SB3 models.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

import config as conf


def run_legacy_tournament():
    """Run legacy Q-learning/Monte-Carlo tournament."""
    from src.game import tournament
    
    print("Running Legacy Q-Learning Tournament...")
    print(f"  Algorithm: {conf.params['algorithm']}")
    print(f"  Iterations: {conf.params['iterations']}")
    
    run = tournament(
        iterations=conf.params['iterations'],
        algo=conf.params['algorithm'],
        comment=conf.params['logging'],
        agent_info=conf.params['model']
    )

    result = pd.concat([
        pd.Series(run[0], name='winner'), 
        pd.Series(run[1], name='turns')
    ], axis=1)
    
    result["win_rate"] = np.where(result["winner"] == conf.player_name_1, 1, 0)
    result["win_rate"] = result["win_rate"].cumsum() / (result.index + 1)

    q_vals = pd.DataFrame(run[2].q)
    q_vals.index.rename("id", inplace=True)

    if not os.path.exists("assets"):
        os.makedirs("assets")

    q_vals.to_csv("assets/q-values.csv", index=True)
    result.to_csv("assets/results.csv", index=False)
    
    final_win_rate = result["win_rate"].iloc[-1]
    print(f"\nFinal Win Rate: {final_win_rate:.2%}")
    return result


def run_sb3_evaluation(model_name=None, num_games=100):
    """Evaluate a Stable Baselines3 model."""
    from src.sb3_agent import UnoEnv
    
    if model_name is None:
        model_name = conf.default_model
    
    if model_name not in conf.sb3_models:
        print(f"Error: Model '{model_name}' not found.")
        print(f"Available models: {list(conf.sb3_models.keys())}")
        return None
    
    model_info = conf.sb3_models[model_name]
    model_path = model_info["path"]
    model_type = model_info["type"]
    
    print(f"\nEvaluating SB3 Model: {model_name}")
    print(f"  Path: {model_path}")
    print(f"  Type: {model_type}")
    print(f"  Games: {num_games}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # Load appropriate model class
    if model_type == "recurrentppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
    elif model_type == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    elif model_type == "dqn":
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    elif model_type == "a2c":
        from stable_baselines3 import A2C
        model = A2C.load(model_path)
    else:
        print(f"Error: Unknown model type '{model_type}'")
        return None
    
    # Run evaluation
    env = UnoEnv()
    wins = 0
    total_turns = []
    
    for game in range(num_games):
        obs, _ = env.reset()
        done = False
        lstm_states = None
        
        if model_type == "recurrentppo":
            episode_starts = np.ones((1,), dtype=bool)
        
        while not done:
            if model_type == "recurrentppo":
                action, lstm_states = model.predict(obs, state=lstm_states, 
                                                    episode_start=episode_starts, 
                                                    deterministic=True)
                episode_starts = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if info.get("winner") == 0:
            wins += 1
        total_turns.append(info.get("turns", 0))
        
        if (game + 1) % 20 == 0:
            print(f"  Progress: {game + 1}/{num_games} games, Win rate: {wins/(game+1):.2%}")
    
    win_rate = wins / num_games
    avg_turns = np.mean(total_turns)
    
    print(f"\n{'='*40}")
    print(f"Results for {model_name}:")
    print(f"  Win Rate: {win_rate:.2%} ({wins}/{num_games})")
    print(f"  Avg Turns: {avg_turns:.1f}")
    print(f"{'='*40}")
    
    return {"model": model_name, "win_rate": win_rate, "wins": wins, 
            "games": num_games, "avg_turns": avg_turns}


def list_models():
    """List all available models."""
    print("\n" + "="*50)
    print("Available Models")
    print("="*50)
    
    print("\n[SB3 Models]")
    for name, info in conf.sb3_models.items():
        default = " (default)" if name == conf.default_model else ""
        print(f"  {name}: {info['description']}{default}")
    
    print("\n[Legacy]")
    print(f"  q-learning: Q-Learning Agent")
    print(f"  monte-carlo: Monte-Carlo Agent")
    print()


def main():
    parser = argparse.ArgumentParser(description="UNO RL - Train and Evaluate Models")
    parser.add_argument("--mode", choices=["legacy", "eval", "list", "gui", "battle"],
                       default="list", help="Run mode")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name for evaluation")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of games for evaluation")
    
    args = parser.parse_args()
    
    if args.mode == "list":
        list_models()
    
    elif args.mode == "legacy":
        run_legacy_tournament()
    
    elif args.mode == "eval":
        run_sb3_evaluation(args.model, args.games)
    
    elif args.mode == "gui":
        print("Launching UNO GUI...")
        os.system(f"{sys.executable} uno_gui.py")
    
    elif args.mode == "battle":
        print("Launching Model Battle Arena...")
        os.system(f"{sys.executable} model_battle_gui.py")


if __name__ == "__main__":
    main()
