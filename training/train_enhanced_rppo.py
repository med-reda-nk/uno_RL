"""
Enhanced Recurrent PPO Training for UNO - Targeting 60%+ Win Rate
Key improvements:
1. Better reward shaping
2. Curriculum learning with opponent difficulty
3. Longer training with more parallel envs
4. Optimized LSTM architecture
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Optional, Dict, Any, Tuple
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("Install: pip install sb3-contrib stable-baselines3[extra] gymnasium torch")
    sys.exit(1)


class EnhancedUnoEnv(gym.Env):
    """
    Enhanced UNO Environment with better reward shaping for higher win rates.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, opponent_strength: float = 0.3, render_mode: Optional[str] = None):
        """
        Args:
            opponent_strength: 0.0 = random, 1.0 = smart opponent
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.opponent_strength = opponent_strength
        
        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)
        
        # Enhanced observation: 21 features
        # - 4: Open card color (one-hot)
        # - 4: Normal cards per color (normalized)
        # - 3: Special cards (SKI, REV, PL2)
        # - 2: Wild cards (PL4, COL)
        # - 4: Playable normal cards per color
        # - 1: Hand size ratio (our cards / (our + opponent))
        # - 1: Opponent hand size (normalized)
        # - 1: Deck size (normalized)
        # - 1: Turn number (normalized)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(21,), dtype=np.float32
        )
        
        self.action_names = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        
        # Import game components
        from src.cards import Deck, Card
        self.Deck = Deck
        self.Card = Card
        
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        """Enhanced observation with game state info."""
        if self.state is None:
            return np.zeros(21, dtype=np.float32)
        
        # One-hot open card color
        color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        color_vec = [0, 0, 0, 0]
        open_color = self.state.get("OPEN", "RED")
        if open_color in color_map:
            color_vec[color_map[open_color]] = 1
        
        # Card counts
        cards = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL",
                 "RED#", "GRE#", "BLU#", "YEL#"]
        
        card_values = []
        for card in cards[:4]:
            card_values.append(self.state.get(card, 0) / 2.0)
        for card in cards[4:9]:
            card_values.append(min(self.state.get(card, 0), 1))
        for card in cards[9:]:
            card_values.append(min(self.state.get(card, 0), 1))
        
        # Additional features
        total_cards = len(self.player_hand) + len(self.opponent_hand)
        hand_ratio = len(self.player_hand) / max(total_cards, 1)
        opp_size = min(len(self.opponent_hand) / 15.0, 1.0)
        deck_size = min(len(self.deck.cards) / 80.0, 1.0)
        turn_norm = min(self.step_count / 100.0, 1.0)
        
        obs = np.array(
            color_vec + card_values + [hand_ratio, opp_size, deck_size, turn_norm],
            dtype=np.float32
        )
        
        # Ensure correct size
        if len(obs) < 21:
            obs = np.pad(obs, (0, 21 - len(obs)))
        elif len(obs) > 21:
            obs = obs[:21]
        
        return obs
    
    def _get_valid_actions_mask(self) -> np.ndarray:
        mask = np.zeros(9, dtype=bool)
        if self.actions_available:
            for action_name, available in self.actions_available.items():
                if available != 0:
                    idx = self.action_names.index(action_name) if action_name in self.action_names else -1
                    if idx >= 0:
                        mask[idx] = True
        return mask
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.deck = self.Deck()
        
        self.card_open = self.deck.draw_from_deck()
        while self.card_open.value not in range(0, 10):
            self.card_open = self.deck.draw_from_deck()
        
        self.player_hand = []
        self.opponent_hand = []
        
        for _ in range(7):
            self.player_hand.append(self.deck.draw_from_deck())
            self.opponent_hand.append(self.deck.draw_from_deck())
        
        self._update_state()
        
        self.done = False
        self.truncated = False
        self.step_count = 0
        self.cards_played = 0
        
        return self._get_obs(), {}
    
    def _update_state(self):
        norm_cards = {"RED": 0, "GRE": 0, "BLU": 0, "YEL": 0}
        spec_cards = {"SKI": 0, "REV": 0, "PL2": 0}
        wild_cards = {"PL4": 0, "COL": 0}
        play_norm = {"RED#": 0, "GRE#": 0, "BLU#": 0, "YEL#": 0}
        play_spec = {"SKI#": 0, "REV#": 0, "PL2#": 0}
        
        self.hand_play = []
        
        for card in self.player_hand:
            if card.color in norm_cards and card.value in range(0, 10):
                norm_cards[card.color] = min(norm_cards[card.color] + 1, 2)
            if card.value in spec_cards:
                spec_cards[card.value] = min(spec_cards[card.value] + 1, 1)
            if card.value in wild_cards:
                wild_cards[card.value] = min(wild_cards[card.value] + 1, 1)
            
            if (card.color == self.card_open.color or 
                card.value == self.card_open.value or 
                card.value in ["COL", "PL4"]):
                self.hand_play.append(card)
                
                if card.color in norm_cards and card.value in range(0, 10):
                    play_norm[card.color + "#"] = 1
                if card.value in spec_cards:
                    play_spec[card.value + "#"] = 1
        
        self.state = {"OPEN": self.card_open.color if self.card_open.color in ["RED", "GRE", "BLU", "YEL"] else "RED"}
        self.state.update(norm_cards)
        self.state.update(spec_cards)
        self.state.update(wild_cards)
        self.state.update(play_norm)
        self.state.update(play_spec)
        
        self.actions_available = {}
        for key in ["RED", "GRE", "BLU", "YEL"]:
            self.actions_available[key] = play_norm.get(key + "#", 0)
        for key in ["SKI", "REV", "PL2"]:
            self.actions_available[key] = play_spec.get(key + "#", 0)
        for key in ["PL4", "COL"]:
            self.actions_available[key] = wild_cards.get(key, 0)
    
    def _find_card_for_action(self, action_name: str):
        for card in self.hand_play:
            if action_name in ["COL", "PL4"] and card.value == action_name:
                return card
            if action_name in ["RED", "GRE", "BLU", "YEL"]:
                if card.color == action_name and card.value in range(0, 10):
                    return card
            if action_name in ["SKI", "REV", "PL2"] and card.value == action_name:
                return card
        return self.hand_play[0] if self.hand_play else None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        reward = 0
        info = {}
        
        valid_mask = self._get_valid_actions_mask()
        action_name = self.action_names[action]
        
        prev_hand_size = len(self.player_hand)
        prev_opp_size = len(self.opponent_hand)
        
        # No playable cards - must draw
        if not self.hand_play:
            if len(self.deck.cards) > 0:
                card = self.deck.draw_from_deck()
                self.player_hand.append(card)
            self._update_state()
            reward = -1.0  # Penalty for drawing
        
        # Invalid action - pick valid one with penalty
        elif not valid_mask[action]:
            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                action_name = self.action_names[action]
                reward = -0.5  # Penalty for invalid action
        
        # Play the card
        if self.hand_play and valid_mask[action]:
            card = self._find_card_for_action(action_name)
            if card:
                self.player_hand.remove(card)
                self.deck.discard(card)
                self.cards_played += 1
                
                # Handle wild cards
                if card.value in ["COL", "PL4"]:
                    colors = [c.color for c in self.player_hand if c.color in ["RED", "GRE", "BLU", "YEL"]]
                    card.color = max(set(colors), key=colors.count) if colors else "RED"
                
                self.card_open = card
                
                # Base reward for playing
                reward += 2.0
                
                # Bonus for special cards
                if card.value in ["SKI", "REV"]:
                    reward += 2.0  # Skip opponent's turn
                elif card.value == "PL2":
                    reward += 3.0
                    for _ in range(2):
                        if len(self.deck.cards) > 0:
                            self.opponent_hand.append(self.deck.draw_from_deck())
                elif card.value == "PL4":
                    reward += 5.0  # Strong card
                    for _ in range(4):
                        if len(self.deck.cards) > 0:
                            self.opponent_hand.append(self.deck.draw_from_deck())
                
                # Bonus for reducing hand size
                reward += 0.5 * (prev_hand_size - len(self.player_hand))
                
                # Bonus for UNO (1 card left)
                if len(self.player_hand) == 1:
                    reward += 5.0
        
        # Check player win
        if len(self.player_hand) == 0:
            self.done = True
            reward += 150  # Increased win reward
            info["winner"] = "player"
            return self._get_obs(), reward, self.done, self.truncated, info
        
        # Opponent turn
        self._opponent_turn()
        
        # Check opponent win
        if len(self.opponent_hand) == 0:
            self.done = True
            reward -= 100  # Increased loss penalty
            info["winner"] = "opponent"
            return self._get_obs(), reward, self.done, self.truncated, info
        
        # Reward for card advantage
        card_diff = prev_opp_size - len(self.opponent_hand) - (prev_hand_size - len(self.player_hand))
        reward += card_diff * 0.3
        
        self._update_state()
        
        # Truncation
        if self.step_count >= 500:
            self.truncated = True
            reward += (len(self.opponent_hand) - len(self.player_hand)) * 3
        
        return self._get_obs(), reward, self.done, self.truncated, info
    
    def _opponent_turn(self):
        """Opponent with adjustable strength."""
        import random
        
        opponent_playable = []
        for card in self.opponent_hand:
            if (card.color == self.card_open.color or 
                card.value == self.card_open.value or 
                card.value in ["COL", "PL4"]):
                opponent_playable.append(card)
        
        if opponent_playable:
            # Smart opponent prioritizes special cards
            if random.random() < self.opponent_strength:
                # Prioritize: PL4 > PL2 > SKI/REV > normal
                priorities = []
                for card in opponent_playable:
                    if card.value == "PL4":
                        priorities.append((card, 4))
                    elif card.value == "PL2":
                        priorities.append((card, 3))
                    elif card.value in ["SKI", "REV"]:
                        priorities.append((card, 2))
                    else:
                        priorities.append((card, 1))
                priorities.sort(key=lambda x: x[1], reverse=True)
                card = priorities[0][0]
            else:
                card = random.choice(opponent_playable)
            
            self.opponent_hand.remove(card)
            self.deck.discard(card)
            
            if card.value in ["COL", "PL4"]:
                # Smart color choice
                if random.random() < self.opponent_strength:
                    colors = [c.color for c in self.opponent_hand if c.color in ["RED", "GRE", "BLU", "YEL"]]
                    card.color = max(set(colors), key=colors.count) if colors else random.choice(["RED", "GRE", "BLU", "YEL"])
                else:
                    card.color = random.choice(["RED", "GRE", "BLU", "YEL"])
            
            self.card_open = card
            
            if card.value == "PL4":
                for _ in range(4):
                    if len(self.deck.cards) > 0:
                        self.player_hand.append(self.deck.draw_from_deck())
            elif card.value == "PL2":
                for _ in range(2):
                    if len(self.deck.cards) > 0:
                        self.player_hand.append(self.deck.draw_from_deck())
        else:
            if len(self.deck.cards) > 0:
                self.opponent_hand.append(self.deck.draw_from_deck())


class WinRateCallback(BaseCallback):
    """Track win rate during training."""
    
    def __init__(self, log_freq=5000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.total_games = 0
        self.best_win_rate = 0
        
    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[i]
                    self.total_games += 1
                    if info.get("winner") == "player":
                        self.wins += 1
        
        if self.n_calls % self.log_freq == 0 and self.total_games > 0:
            win_rate = self.wins / self.total_games * 100
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
            
            if self.verbose > 0:
                print(f"Step {self.n_calls:,}: Games={self.total_games}, "
                      f"WinRate={win_rate:.1f}%, Best={self.best_win_rate:.1f}%")
        
        return True


def cosine_schedule(initial_value: float, min_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + (initial_value - min_value) * 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
    return func


def make_env(rank, seed=0, opponent_strength=0.3):
    def _init():
        env = EnhancedUnoEnv(opponent_strength=opponent_strength)
        env = Monitor(env)
        return env
    set_random_seed(seed + rank)
    return _init


def evaluate_model(model, n_episodes=300, opponent_strength=0.3) -> dict:
    """Evaluate the model."""
    env = EnhancedUnoEnv(opponent_strength=opponent_strength)
    
    wins = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
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
            
    return {
        "wins": wins,
        "losses": n_episodes - wins,
        "win_rate": wins / n_episodes * 100,
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_episode_length": np.mean(episode_lengths),
    }


def train_enhanced_rppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 16,
    opponent_strength: float = 0.3,
    save_path: str = "models/enhanced_rppo",
    seed: int = 42,
):
    """Train enhanced Recurrent PPO for maximum win rate."""
    
    print(f"\n{'='*70}")
    print("🚀 ENHANCED Recurrent PPO Training for UNO (Targeting 60%+ Win Rate)")
    print(f"{'='*70}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Opponent Strength: {opponent_strength}")
    print(f"{'='*70}\n")
    
    # Create environments
    print(f"Creating {n_envs} parallel environments...")
    env = DummyVecEnv([make_env(i, seed, opponent_strength) for i in range(n_envs)])
    
    env = VecNormalize(
        env, 
        norm_obs=False,
        norm_reward=True,
        clip_reward=15.0,
        gamma=0.995,
    )
    
    # Eval env
    eval_env = DummyVecEnv([make_env(i, seed + 1000, opponent_strength) for i in range(4)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        clip_reward=15.0,
        gamma=0.995,
        training=False,
    )
    
    # Optimized config for higher win rate
    n_steps = 512  # Longer sequences
    batch_size = 128
    
    rppo_config = {
        "learning_rate": cosine_schedule(3e-4, min_value=5e-6),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 15,  # More epochs per update
        "gamma": 0.995,  # Higher for long-term planning
        "gae_lambda": 0.98,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,  # Lower entropy for exploitation
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        
        "policy_kwargs": dict(
            lstm_hidden_size=512,  # Larger LSTM
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(
                pi=[512, 256],  # Larger networks
                vf=[512, 256],
            ),
            ortho_init=True,
        ),
        
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": "./logs/enhanced_rppo/",
        "device": "auto",
    }
    
    print("Enhanced Configuration:")
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  LSTM hidden size: 512")
    print(f"  Policy network: [512, 256]")
    print(f"  Learning rate: 3e-4 -> 5e-6 (cosine)")
    print(f"  Epochs per update: 15")
    print()
    
    model = RecurrentPPO("MlpLstmPolicy", env, **rppo_config)
    
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    callbacks = [
        WinRateCallback(log_freq=10000, verbose=1),
        CheckpointCallback(
            save_freq=max(100000 // n_envs, 2000),
            save_path="./models/checkpoints/",
            name_prefix="enhanced_rppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="./models/best_enhanced_rppo/",
            log_path="./logs/enhanced_eval/",
            eval_freq=max(50000 // n_envs, 1000),
            n_eval_episodes=50,
            deterministic=True,
        ),
    ]
    
    print(f"🚀 Starting training for {total_timesteps:,} timesteps...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted. Saving model...")
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl")
    
    print(f"\n✅ Model saved to {save_path}.zip")
    print(f"⏱️ Training time: {training_time}")
    
    # Evaluate
    print("\n📊 Running evaluation (500 episodes)...")
    stats = evaluate_model(model, n_episodes=500, opponent_strength=opponent_strength)
    
    print(f"\n{'='*50}")
    print("📊 FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Wins: {stats['wins']} / 500")
    print(f"Average Reward: {stats['avg_reward']:.2f}")
    print(f"{'='*50}\n")
    
    # Save results
    stats["model_name"] = "Enhanced Recurrent PPO"
    stats["algorithm"] = "ENHANCED_RPPO"
    stats["model_path"] = f"{save_path}.zip"
    stats["timesteps"] = total_timesteps
    stats["training_time"] = str(training_time)
    stats["opponent_strength"] = opponent_strength
    stats["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    results_dir = "comparison_results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "enhanced_rppo_results.csv")
    
    df = pd.DataFrame([stats])
    df.to_csv(csv_path, index=False)
    print(f"📄 Results saved to {csv_path}")
    
    return model, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Enhanced Recurrent PPO for UNO")
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000,
                        help="Total timesteps (default: 1,000,000)")
    parser.add_argument("--envs", "-e", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--opponent", "-o", type=float, default=0.3,
                        help="Opponent strength 0-1 (default: 0.3)")
    parser.add_argument("--save-path", "-s", type=str, default="models/enhanced_rppo",
                        help="Path to save model")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    model, stats = train_enhanced_rppo(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        opponent_strength=args.opponent,
        save_path=args.save_path,
        seed=args.seed,
    )
