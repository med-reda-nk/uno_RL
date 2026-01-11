"""
UNO RL Agent using Stable Baselines3
Uses pre-built, production-ready RL algorithms.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces

# Try to import Stable Baselines3
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Stable Baselines3 not installed. Run: pip install stable-baselines3[extra]")


class UnoEnv(gym.Env):
    """
    Custom Gymnasium Environment for UNO game.
    Compatible with Stable Baselines3 algorithms.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Action space: 9 discrete actions
        # 0-3: Play colored card (RED, GRE, BLU, YEL)
        # 4-6: Play special card (SKI, REV, PL2)
        # 7-8: Play wild card (PL4, COL)
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 17 features
        # - 4: One-hot encoded open card color
        # - 4: Number of normal cards per color (0-2, normalized)
        # - 3: Number of special cards (SKI, REV, PL2) (0-1)
        # - 2: Number of wild cards (PL4, COL) (0-1)
        # - 4: Number of playable normal cards per color (0-1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(17,), dtype=np.float32
        )
        
        # Action mapping
        self.action_names = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        
        # Game state
        self.state = None
        self.actions_available = None
        self.done = False
        self.truncated = False
        self.step_count = 0
        self.max_steps = 500
        
        # Import game components
        from src.cards import Deck, Card
        from src.players import Player
        self.Deck = Deck
        self.Card = Card
        
        # Initialize game
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        """Convert current state to observation vector."""
        if self.state is None:
            return np.zeros(17, dtype=np.float32)
        
        # One-hot encode open card color
        color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        color_vec = [0, 0, 0, 0]
        open_color = self.state.get("OPEN", "RED")
        if open_color in color_map:
            color_vec[color_map[open_color]] = 1
        
        # Card counts (normalized)
        cards = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL",
                 "RED#", "GRE#", "BLU#", "YEL#"]
        
        card_values = []
        for card in cards[:4]:  # Normal cards (max 2)
            card_values.append(self.state.get(card, 0) / 2.0)
        for card in cards[4:9]:  # Special/wild cards (max 1-2)
            card_values.append(min(self.state.get(card, 0), 1))
        for card in cards[9:]:  # Playable cards
            card_values.append(min(self.state.get(card, 0), 1))
        
        obs = np.array(color_vec + card_values, dtype=np.float32)
        
        # Pad or truncate to correct size
        if len(obs) < 17:
            obs = np.pad(obs, (0, 17 - len(obs)))
        elif len(obs) > 17:
            obs = obs[:17]
        
        return obs
    
    def _get_valid_actions_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        mask = np.zeros(9, dtype=bool)
        if self.actions_available:
            for action_name, available in self.actions_available.items():
                if available != 0:
                    idx = self.action_names.index(action_name) if action_name in self.action_names else -1
                    if idx >= 0:
                        mask[idx] = True
        return mask
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize deck and players
        self.deck = self.Deck()
        
        # Get initial open card (must be a normal card)
        self.card_open = self.deck.draw_from_deck()
        while self.card_open.value not in range(0, 10):
            self.card_open = self.deck.draw_from_deck()
        
        # Initialize player hands
        self.player_hand = []
        self.opponent_hand = []
        
        for _ in range(7):
            self.player_hand.append(self.deck.draw_from_deck())
            self.opponent_hand.append(self.deck.draw_from_deck())
        
        # Update state
        self._update_state()
        
        self.done = False
        self.truncated = False
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def _update_state(self):
        """Update the state representation based on current hand."""
        norm_cards = {"RED": 0, "GRE": 0, "BLU": 0, "YEL": 0}
        spec_cards = {"SKI": 0, "REV": 0, "PL2": 0}
        wild_cards = {"PL4": 0, "COL": 0}
        play_norm = {"RED#": 0, "GRE#": 0, "BLU#": 0, "YEL#": 0}
        play_spec = {"SKI#": 0, "REV#": 0, "PL2#": 0}
        
        self.hand_play = []
        
        for card in self.player_hand:
            # Count cards by type
            if card.color in norm_cards and card.value in range(0, 10):
                norm_cards[card.color] = min(norm_cards[card.color] + 1, 2)
            if card.value in spec_cards:
                spec_cards[card.value] = min(spec_cards[card.value] + 1, 1)
            if card.value in wild_cards:
                wild_cards[card.value] = min(wild_cards[card.value] + 1, 1)
            
            # Check if card is playable
            if (card.color == self.card_open.color or 
                card.value == self.card_open.value or 
                card.value in ["COL", "PL4"]):
                self.hand_play.append(card)
                
                if card.color in norm_cards and card.value in range(0, 10):
                    play_norm[card.color + "#"] = 1
                if card.value in spec_cards:
                    play_spec[card.value + "#"] = 1
        
        # Build state dictionary
        self.state = {"OPEN": self.card_open.color if self.card_open.color in ["RED", "GRE", "BLU", "YEL"] else "RED"}
        self.state.update(norm_cards)
        self.state.update(spec_cards)
        self.state.update(wild_cards)
        self.state.update(play_norm)
        self.state.update(play_spec)
        
        # Build actions dictionary
        self.actions_available = {}
        for key in ["RED", "GRE", "BLU", "YEL"]:
            self.actions_available[key] = play_norm.get(key + "#", 0)
        for key in ["SKI", "REV", "PL2"]:
            self.actions_available[key] = play_spec.get(key + "#", 0)
        for key in ["PL4", "COL"]:
            self.actions_available[key] = wild_cards.get(key, 0)
    
    def _find_card_for_action(self, action_name: str):
        """Find a card in hand that matches the action."""
        for card in self.hand_play:
            if action_name in ["COL", "PL4"] and card.value == action_name:
                return card
            if action_name in ["RED", "GRE", "BLU", "YEL"]:
                if card.color == action_name and card.value in range(0, 10):
                    return card
            if action_name in ["SKI", "REV", "PL2"] and card.value == action_name:
                return card
        
        # Fallback: return any playable card
        return self.hand_play[0] if self.hand_play else None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Parameters:
        -----------
        action : int
            The action to take (0-8)
        
        Returns:
        --------
        observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        reward = 0
        info = {}
        
        # Check if action is valid
        valid_mask = self._get_valid_actions_mask()
        action_name = self.action_names[action]
        
        # If no playable cards, draw
        if not self.hand_play:
            if len(self.deck.cards) > 0:
                card = self.deck.draw_from_deck()
                self.player_hand.append(card)
            self._update_state()
            reward = -0.5  # Small penalty for drawing
        
        # If action is invalid, pick a valid one
        elif not valid_mask[action]:
            valid_actions = np.where(valid_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                action_name = self.action_names[action]
                reward = -0.2  # Small penalty for invalid action
        
        # Play the card
        if self.hand_play and valid_mask[action]:
            card = self._find_card_for_action(action_name)
            if card:
                self.player_hand.remove(card)
                self.deck.discard(card)
                
                # Handle wild cards
                if card.value in ["COL", "PL4"]:
                    colors = [c.color for c in self.player_hand if c.color in ["RED", "GRE", "BLU", "YEL"]]
                    card.color = max(set(colors), key=colors.count) if colors else "RED"
                
                self.card_open = card
                reward += 1.0
                
                # Bonus for special cards
                if card.value in ["SKI", "REV", "PL2"]:
                    reward += 1.0
                elif card.value == "PL4":
                    reward += 2.0
                    # Opponent draws 4
                    for _ in range(4):
                        if len(self.deck.cards) > 0:
                            self.opponent_hand.append(self.deck.draw_from_deck())
                elif card.value == "PL2":
                    # Opponent draws 2
                    for _ in range(2):
                        if len(self.deck.cards) > 0:
                            self.opponent_hand.append(self.deck.draw_from_deck())
        
        # Check for player win
        if len(self.player_hand) == 0:
            self.done = True
            reward += 100
            info["winner"] = "player"
            return self._get_obs(), reward, self.done, self.truncated, info
        
        # Opponent turn (random policy)
        self._opponent_turn()
        
        # Check for opponent win
        if len(self.opponent_hand) == 0:
            self.done = True
            reward -= 50
            info["winner"] = "opponent"
            return self._get_obs(), reward, self.done, self.truncated, info
        
        # Update state after all moves
        self._update_state()
        
        # Truncate if too many steps
        if self.step_count >= self.max_steps:
            self.truncated = True
            # Partial reward based on hand size
            reward += (len(self.opponent_hand) - len(self.player_hand)) * 2
        
        return self._get_obs(), reward, self.done, self.truncated, info
    
    def _opponent_turn(self):
        """Execute opponent's turn with random policy."""
        import random
        
        # Find playable cards
        opponent_playable = []
        for card in self.opponent_hand:
            if (card.color == self.card_open.color or 
                card.value == self.card_open.value or 
                card.value in ["COL", "PL4"]):
                opponent_playable.append(card)
        
        if opponent_playable:
            card = random.choice(opponent_playable)
            self.opponent_hand.remove(card)
            self.deck.discard(card)
            
            if card.value in ["COL", "PL4"]:
                card.color = random.choice(["RED", "GRE", "BLU", "YEL"])
            
            self.card_open = card
            
            # Handle special cards
            if card.value == "PL4":
                for _ in range(4):
                    if len(self.deck.cards) > 0:
                        self.player_hand.append(self.deck.draw_from_deck())
            elif card.value == "PL2":
                for _ in range(2):
                    if len(self.deck.cards) > 0:
                        self.player_hand.append(self.deck.draw_from_deck())
        else:
            # Draw a card
            if len(self.deck.cards) > 0:
                self.opponent_hand.append(self.deck.draw_from_deck())
    
    def render(self):
        """Render the current game state."""
        if self.render_mode == "human":
            print(f"\n--- UNO Game State ---")
            print(f"Open Card: {self.card_open.color} {self.card_open.value}")
            print(f"Your Hand ({len(self.player_hand)} cards): ", end="")
            for card in self.player_hand:
                print(f"[{card.color} {card.value}]", end=" ")
            print(f"\nOpponent: {len(self.opponent_hand)} cards")
            print(f"Deck: {len(self.deck.cards)} cards")


class SB3UnoAgent:
    """
    Wrapper for Stable Baselines3 agents to play UNO.
    Supports DQN, PPO, and A2C algorithms.
    """
    
    ALGORITHMS = {
        "dqn": DQN if SB3_AVAILABLE else None,
        "ppo": PPO if SB3_AVAILABLE else None,
        "a2c": A2C if SB3_AVAILABLE else None,
    }
    
    def __init__(self, algorithm: str = "dqn", config: Optional[Dict] = None):
        """
        Initialize the SB3 agent.
        
        Parameters:
        -----------
        algorithm : str
            RL algorithm to use: "dqn", "ppo", or "a2c"
        config : dict
            Configuration parameters for the algorithm
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines3 is required. Install with: pip install stable-baselines3[extra]")
        
        self.algorithm_name = algorithm.lower()
        self.config = config or {}
        
        # Create environment
        self.env = UnoEnv()
        self.env = Monitor(self.env)
        
        # Get algorithm class
        algo_class = self.ALGORITHMS.get(self.algorithm_name)
        if algo_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(self.ALGORITHMS.keys())}")
        
        # Default configurations
        if self.algorithm_name == "dqn":
            default_config = {
                "learning_rate": 1e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 64,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "policy_kwargs": dict(net_arch=[256, 256, 128]),
                "verbose": 1,
            }
        elif self.algorithm_name == "ppo":
            default_config = {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                "verbose": 1,
            }
        else:  # a2c
            default_config = {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                "verbose": 1,
            }
        
        # Merge with user config
        default_config.update(self.config)
        
        # Create model
        self.model = algo_class("MlpPolicy", self.env, **default_config)
        
        # For compatibility with existing game code
        self.prev_state = None
        self.prev_action = None
        self.epsilon = 0.0  # SB3 handles exploration internally
    
    def train(self, total_timesteps: int = 100000, save_path: Optional[str] = None,
              eval_freq: int = 10000, n_eval_episodes: int = 10):
        """
        Train the agent.
        
        Parameters:
        -----------
        total_timesteps : int
            Total number of environment steps to train
        save_path : str
            Path to save the trained model
        eval_freq : int
            Evaluate every N steps
        n_eval_episodes : int
            Number of episodes for evaluation
        """
        print(f"\n{'='*60}")
        print(f"Training {self.algorithm_name.upper()} Agent with Stable Baselines3")
        print(f"{'='*60}\n")
        
        callbacks = []
        
        # Evaluation callback
        if eval_freq > 0:
            eval_env = Monitor(UnoEnv())
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="./models/" if save_path else None,
                log_path="./logs/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "models", exist_ok=True)
            self.model.save(save_path)
            print(f"\nModel saved to {save_path}")
        
        return self
    
    def load(self, path: str):
        """Load a trained model."""
        algo_class = self.ALGORITHMS[self.algorithm_name]
        self.model = algo_class.load(path, env=self.env)
        print(f"Model loaded from {path}")
        return self
    
    def save(self, path: str = "models/sb3_uno_agent"):
        """Save the model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else "models", exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Get action from observation."""
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def step(self, state_dict: Dict, actions_dict: Dict) -> str:
        """
        Compatible interface with existing game code.
        
        Parameters:
        -----------
        state_dict : dict
            Current state representation
        actions_dict : dict
            Available actions
        
        Returns:
        --------
        str : Action name
        """
        # Convert state to observation
        obs = self._state_dict_to_obs(state_dict)
        
        # Get action from model
        action = self.predict(obs, deterministic=False)
        
        # Map to action name
        action_names = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        action_name = action_names[action]
        
        # If action is invalid, choose a valid one
        valid_actions = [k for k, v in actions_dict.items() if v != 0]
        if action_name not in valid_actions and valid_actions:
            import random
            action_name = random.choice(valid_actions)
        
        self.prev_state = state_dict
        self.prev_action = action_name
        
        return action_name
    
    def _state_dict_to_obs(self, state_dict: Dict) -> np.ndarray:
        """Convert state dictionary to observation vector."""
        color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        color_vec = [0, 0, 0, 0]
        if state_dict.get("OPEN") in color_map:
            color_vec[color_map[state_dict["OPEN"]]] = 1
        
        cards = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL",
                 "RED#", "GRE#", "BLU#", "YEL#"]
        
        card_values = []
        for card in cards[:4]:
            card_values.append(state_dict.get(card, 0) / 2.0)
        for card in cards[4:9]:
            card_values.append(min(state_dict.get(card, 0), 1))
        for card in cards[9:]:
            card_values.append(min(state_dict.get(card, 0), 1))
        
        obs = np.array(color_vec + card_values, dtype=np.float32)
        
        if len(obs) < 17:
            obs = np.pad(obs, (0, 17 - len(obs)))
        elif len(obs) > 17:
            obs = obs[:17]
        
        return obs
    
    def update(self, state_dict: Dict, action: str):
        """Compatibility method - SB3 handles updates internally during training."""
        pass
    
    def end_episode(self, won: bool):
        """Compatibility method."""
        self.prev_state = None
        self.prev_action = None
    
    def evaluate(self, n_episodes: int = 100) -> Dict:
        """
        Evaluate the agent.
        
        Parameters:
        -----------
        n_episodes : int
            Number of episodes to evaluate
        
        Returns:
        --------
        dict : Evaluation metrics
        """
        from tqdm import tqdm
        
        wins = 0
        total_reward = 0
        episode_lengths = []
        
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            obs, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            steps = 0
            
            while not done and not truncated:
                action = self.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            if info.get("winner") == "player":
                wins += 1
            
            total_reward += episode_reward
            episode_lengths.append(steps)
        
        metrics = {
            "n_episodes": n_episodes,
            "wins": wins,
            "win_rate": wins / n_episodes,
            "avg_reward": total_reward / n_episodes,
            "avg_episode_length": np.mean(episode_lengths),
        }
        
        print(f"\n{'='*40}")
        print("Evaluation Results")
        print(f"{'='*40}")
        print(f"Episodes: {n_episodes}")
        print(f"Wins: {wins} ({metrics['win_rate']:.2%})")
        print(f"Avg Reward: {metrics['avg_reward']:.2f}")
        print(f"Avg Episode Length: {metrics['avg_episode_length']:.1f}")
        
        return metrics


class ProgressCallback(BaseCallback):
    """Custom callback for training progress."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode info
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]
            self.episode_rewards.append(info.get("episode", {}).get("r", 0))
            self.episode_lengths.append(info.get("episode", {}).get("l", 0))
        return True


def create_agent(algorithm: str = "dqn", config: Optional[Dict] = None) -> SB3UnoAgent:
    """
    Factory function to create an SB3 agent.
    
    Parameters:
    -----------
    algorithm : str
        Algorithm to use: "dqn", "ppo", or "a2c"
    config : dict
        Configuration parameters
    
    Returns:
    --------
    SB3UnoAgent
    """
    return SB3UnoAgent(algorithm=algorithm, config=config)


if __name__ == "__main__":
    # Quick test
    if SB3_AVAILABLE:
        print("Testing Stable Baselines3 UNO Agent...")
        
        # Create and train a DQN agent
        agent = create_agent("dqn")
        agent.train(total_timesteps=10000, save_path="models/sb3_dqn_uno")
        
        # Evaluate
        agent.evaluate(n_episodes=50)
    else:
        print("Please install Stable Baselines3: pip install stable-baselines3[extra]")
