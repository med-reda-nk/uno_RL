"""
Multiplayer UNO Environment (2-4 Players)
=========================================
Supports 2, 3, or 4 players with flexible opponent types:
- Random opponents
- Trained AI opponents (self-play)
- Mixed opponents

This environment enables:
1. Training against multiple opponents
2. Self-play training for improved performance
3. Curriculum learning (easy -> hard opponents)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from gymnasium import spaces
import random

# Import game components
from src.cards import Deck, Card


class MultiplayerUnoEnv(gym.Env):
    """
    Multiplayer UNO Environment supporting 2-4 players.
    
    Features:
    - Variable number of players (2-4)
    - Self-play capability with loaded models
    - Curriculum learning support
    - Enhanced observation space with opponent info
    
    Observation Space (25 features for 4-player):
        - 4: One-hot encoded open card color
        - 4: Number of normal cards per color (normalized)
        - 3: Number of special cards (SKI, REV, PL2)
        - 2: Number of wild cards (PL4, COL)
        - 4: Playable normal cards per color
        - 3: Number of cards each opponent has (normalized, max 3 opponents)
        - 1: Current player position in turn order
        - 1: Direction (1 for clockwise, 0 for counter-clockwise)
        - 1: Cards remaining in deck (normalized)
        - 2: Game progress indicators
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        num_players: int = 4,
        render_mode: Optional[str] = None,
        opponent_models: Optional[List] = None,
        curriculum_level: float = 0.0,  # 0.0 = all random, 1.0 = all trained
    ):
        """
        Initialize multiplayer UNO environment.
        
        Args:
            num_players: Number of players (2-4)
            render_mode: Rendering mode ('human' or 'rgb_array')
            opponent_models: List of trained models for opponents (or None for random)
            curriculum_level: Probability of using trained opponent vs random (0.0-1.0)
        """
        super().__init__()
        
        assert 2 <= num_players <= 4, "Number of players must be between 2 and 4"
        
        self.num_players = num_players
        self.render_mode = render_mode
        self.opponent_models = opponent_models or []
        self.curriculum_level = curriculum_level
        
        # Action space: 9 discrete actions (same as 2-player)
        # 0-3: Play colored card (RED, GRE, BLU, YEL)
        # 4-6: Play special card (SKI, REV, PL2)
        # 7-8: Play wild card (PL4, COL)
        self.action_space = spaces.Discrete(9)
        
        # Enhanced observation space (25 features)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(25,), dtype=np.float32
        )
        
        # Action and color mappings
        self.action_names = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        self.color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        
        # Game state
        self.deck: Optional[Deck] = None
        self.hands: List[List[Card]] = []  # hands[0] is the RL agent
        self.card_open: Optional[Card] = None
        self.current_player: int = 0
        self.direction: int = 1  # 1 = clockwise, -1 = counter-clockwise
        self.done: bool = False
        self.truncated: bool = False
        self.step_count: int = 0
        self.max_steps: int = 500
        self.winner: Optional[int] = None
        
        # State tracking
        self.state: Dict = {}
        self.actions_available: Dict = {}
        self.hand_play: List[Card] = []
        
        # Statistics
        self.games_played = 0
        self.wins = 0
        
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        """Convert current state to observation vector (25 features)."""
        obs = np.zeros(25, dtype=np.float32)
        
        if self.card_open is None:
            return obs
        
        # Feature 0-3: One-hot encoded open card color
        open_color = self.card_open.color
        if open_color in self.color_map:
            obs[self.color_map[open_color]] = 1.0
        
        # Feature 4-7: Number of normal cards per color (normalized by max 2)
        for i, color in enumerate(["RED", "GRE", "BLU", "YEL"]):
            count = sum(1 for c in self.hands[0] if c.color == color and isinstance(c.value, int))
            obs[4 + i] = min(count / 2.0, 1.0)
        
        # Feature 8-10: Number of special cards
        for i, val in enumerate(["SKI", "REV", "PL2"]):
            count = sum(1 for c in self.hands[0] if c.value == val)
            obs[8 + i] = min(count, 1.0)
        
        # Feature 11-12: Number of wild cards
        for i, val in enumerate(["PL4", "COL"]):
            count = sum(1 for c in self.hands[0] if c.value == val)
            obs[11 + i] = min(count, 1.0)
        
        # Feature 13-16: Playable normal cards per color (binary)
        for i, color in enumerate(["RED", "GRE", "BLU", "YEL"]):
            has_playable = any(
                c.color == color and isinstance(c.value, int) and self._is_playable(c)
                for c in self.hands[0]
            )
            obs[13 + i] = 1.0 if has_playable else 0.0
        
        # Feature 17-19: Opponent hand sizes (normalized by 15 cards)
        for i in range(3):
            if i + 1 < self.num_players:
                obs[17 + i] = min(len(self.hands[i + 1]) / 15.0, 1.0)
        
        # Feature 20: Current position in turn order (normalized)
        obs[20] = self.current_player / (self.num_players - 1) if self.num_players > 1 else 0
        
        # Feature 21: Direction (1 for clockwise, 0 for counter)
        obs[21] = 1.0 if self.direction == 1 else 0.0
        
        # Feature 22: Cards remaining in deck (normalized by 108)
        obs[22] = len(self.deck.cards) / 108.0 if self.deck else 0.0
        
        # Feature 23: Minimum opponent hand size (normalized) - threat level
        if self.num_players > 1:
            min_opponent_cards = min(len(self.hands[i]) for i in range(1, self.num_players))
            obs[23] = 1.0 - min(min_opponent_cards / 7.0, 1.0)  # Higher = more threat
        
        # Feature 24: Our hand size relative to start (normalized)
        obs[24] = min(len(self.hands[0]) / 7.0, 1.0)
        
        return obs
    
    def _is_playable(self, card: Card) -> bool:
        """Check if a card can be played on the current open card."""
        if self.card_open is None:
            return True
        return (
            card.color == self.card_open.color or
            card.value == self.card_open.value or
            card.value in ["COL", "PL4"]
        )
    
    def _get_valid_actions_mask(self) -> np.ndarray:
        """Get mask of valid actions for the current player."""
        mask = np.zeros(9, dtype=bool)
        
        for card in self.hands[0]:
            if not self._is_playable(card):
                continue
            
            # Map card to action
            if card.value in ["COL", "PL4"]:
                idx = self.action_names.index(card.value)
            elif card.value in ["SKI", "REV", "PL2"]:
                idx = self.action_names.index(card.value)
            elif card.color in self.color_map and isinstance(card.value, int):
                idx = self.color_map[card.color]
            else:
                continue
            
            mask[idx] = True
        
        return mask
    
    def _update_playable_cards(self):
        """Update list of playable cards for the RL agent."""
        self.hand_play = [c for c in self.hands[0] if self._is_playable(c)]
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to start a new game."""
        super().reset(seed=seed)
        
        # Initialize deck
        self.deck = Deck()
        
        # Initialize hands for all players
        self.hands = [[] for _ in range(self.num_players)]
        for _ in range(7):
            for player_idx in range(self.num_players):
                self.hands[player_idx].append(self.deck.draw_from_deck())
        
        # Draw open card (must be a number card)
        self.card_open = self.deck.draw_from_deck()
        while not isinstance(self.card_open.value, int):
            self.deck.discard(self.card_open)
            self.card_open = self.deck.draw_from_deck()
        
        # Reset game state
        self.current_player = 0
        self.direction = 1
        self.done = False
        self.truncated = False
        self.step_count = 0
        self.winner = None
        
        self._update_playable_cards()
        
        return self._get_obs(), {"num_players": self.num_players}
    
    def _find_card_for_action(self, action: int) -> Optional[Card]:
        """Find a playable card matching the given action."""
        action_name = self.action_names[action]
        
        for card in self.hand_play:
            if action_name in ["COL", "PL4"] and card.value == action_name:
                return card
            if action_name in ["RED", "GRE", "BLU", "YEL"]:
                if card.color == action_name and isinstance(card.value, int):
                    return card
            if action_name in ["SKI", "REV", "PL2"] and card.value == action_name:
                return card
        
        return self.hand_play[0] if self.hand_play else None
    
    def _get_next_player(self) -> int:
        """Get the next player in turn order."""
        return (self.current_player + self.direction) % self.num_players
    
    def _apply_card_effects(self, card: Card, played_by: int) -> float:
        """Apply special card effects and return bonus reward."""
        bonus = 0.0
        next_player = (played_by + self.direction) % self.num_players
        
        if card.value == "SKI":
            # Skip next player - they lose their turn
            self.current_player = (next_player + self.direction) % self.num_players
            bonus = 1.5 if played_by == 0 else -0.5
            
        elif card.value == "REV":
            # Reverse direction
            self.direction *= -1
            # In 2-player, reverse acts like skip
            if self.num_players == 2:
                self.current_player = played_by
            bonus = 1.0 if played_by == 0 else -0.3
            
        elif card.value == "PL2":
            # Next player draws 2 and loses turn
            for _ in range(2):
                if self.deck.cards:
                    self.hands[next_player].append(self.deck.draw_from_deck())
            self.current_player = (next_player + self.direction) % self.num_players
            bonus = 2.0 if played_by == 0 else -1.0
            
        elif card.value == "PL4":
            # Next player draws 4 and loses turn
            for _ in range(4):
                if self.deck.cards:
                    self.hands[next_player].append(self.deck.draw_from_deck())
            self.current_player = (next_player + self.direction) % self.num_players
            bonus = 3.0 if played_by == 0 else -2.0
            
        elif card.value == "COL":
            # Just changes color, no other effect
            bonus = 0.5 if played_by == 0 else 0.0
        
        # Handle wild card color selection
        if card.value in ["COL", "PL4"]:
            if played_by == 0:
                # RL agent picks most common color in hand
                colors = [c.color for c in self.hands[0] if c.color in ["RED", "GRE", "BLU", "YEL"]]
                card.color = max(set(colors), key=colors.count) if colors else "RED"
            else:
                # Opponent picks random color
                card.color = random.choice(["RED", "GRE", "BLU", "YEL"])
        
        return bonus
    
    def _opponent_turn(self, player_idx: int) -> bool:
        """
        Execute an opponent's turn.
        
        Args:
            player_idx: Index of the opponent (1 to num_players-1)
        
        Returns:
            True if the opponent won, False otherwise
        """
        # Find playable cards
        playable = [c for c in self.hands[player_idx] if self._is_playable(c)]
        
        if playable:
            # Use trained model or random based on curriculum level
            use_trained = (
                self.opponent_models and 
                player_idx - 1 < len(self.opponent_models) and
                random.random() < self.curriculum_level
            )
            
            if use_trained:
                # Use trained model for action selection
                model = self.opponent_models[player_idx - 1]
                obs = self._get_opponent_obs(player_idx)
                action, _ = model.predict(obs, deterministic=True)
                card = self._find_opponent_card(player_idx, action)
                if card is None:
                    card = random.choice(playable)
            else:
                # Random selection with slight preference for special cards
                special = [c for c in playable if c.value in ["SKI", "REV", "PL2", "PL4"]]
                if special and random.random() < 0.6:
                    card = random.choice(special)
                else:
                    card = random.choice(playable)
            
            # Play the card
            self.hands[player_idx].remove(card)
            self.deck.discard(card)
            self._apply_card_effects(card, player_idx)
            self.card_open = card
            
            # Check for win
            if len(self.hands[player_idx]) == 0:
                return True
        else:
            # Draw a card
            if self.deck.cards:
                self.hands[player_idx].append(self.deck.draw_from_deck())
        
        return False
    
    def _get_opponent_obs(self, player_idx: int) -> np.ndarray:
        """Get observation from opponent's perspective (for self-play)."""
        # Temporarily swap hands to get observation
        temp = self.hands[0]
        self.hands[0] = self.hands[player_idx]
        obs = self._get_obs()
        self.hands[0] = temp
        return obs
    
    def _find_opponent_card(self, player_idx: int, action: int) -> Optional[Card]:
        """Find a card for opponent based on action."""
        action_name = self.action_names[action]
        playable = [c for c in self.hands[player_idx] if self._is_playable(c)]
        
        for card in playable:
            if action_name in ["COL", "PL4"] and card.value == action_name:
                return card
            if action_name in ["RED", "GRE", "BLU", "YEL"]:
                if card.color == action_name and isinstance(card.value, int):
                    return card
            if action_name in ["SKI", "REV", "PL2"] and card.value == action_name:
                return card
        
        return None
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        The RL agent (player 0) takes an action, then all opponents take their turns.
        """
        self.step_count += 1
        reward = 0.0
        info = {}
        
        # Get valid actions
        valid_mask = self._get_valid_actions_mask()
        self._update_playable_cards()
        
        # RL Agent's turn
        if not self.hand_play:
            # No playable cards - must draw
            if self.deck.cards:
                self.hands[0].append(self.deck.draw_from_deck())
            reward -= 0.3
        else:
            # Check if action is valid
            if not valid_mask[action]:
                # Invalid action - pick a valid one and penalize
                valid_actions = np.where(valid_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                    reward -= 0.15
            
            # Play the card
            card = self._find_card_for_action(action)
            if card:
                self.hands[0].remove(card)
                self.deck.discard(card)
                
                # Base reward for playing
                reward += 0.8
                
                # Apply card effects and get bonus
                bonus = self._apply_card_effects(card, 0)
                reward += bonus
                
                self.card_open = card
                
                # Check for RL agent win
                if len(self.hands[0]) == 0:
                    self.done = True
                    self.winner = 0
                    self.wins += 1
                    self.games_played += 1
                    
                    # Big reward for winning, scaled by number of players
                    reward += 50 * (self.num_players - 1)
                    info["winner"] = "agent"
                    info["win_rate"] = self.wins / self.games_played if self.games_played > 0 else 0
                    return self._get_obs(), reward, self.done, self.truncated, info
        
        # Opponents' turns
        for opponent_idx in range(1, self.num_players):
            if self._opponent_turn(opponent_idx):
                # Opponent won
                self.done = True
                self.winner = opponent_idx
                self.games_played += 1
                
                # Penalty scaled by how close we were to winning
                cards_left = len(self.hands[0])
                reward -= 30 + cards_left * 2
                
                info["winner"] = f"opponent_{opponent_idx}"
                info["cards_left"] = cards_left
                return self._get_obs(), reward, self.done, self.truncated, info
        
        # Update state
        self._update_playable_cards()
        
        # Small reward for having fewer cards
        cards_diff = sum(len(self.hands[i]) for i in range(1, self.num_players)) / (self.num_players - 1) - len(self.hands[0])
        reward += cards_diff * 0.05
        
        # Truncate if too many steps
        if self.step_count >= self.max_steps:
            self.truncated = True
            self.games_played += 1
            
            # Partial reward based on hand sizes
            our_cards = len(self.hands[0])
            avg_opponent = sum(len(self.hands[i]) for i in range(1, self.num_players)) / (self.num_players - 1)
            reward += (avg_opponent - our_cards) * 3
            
            info["truncated_reason"] = "max_steps"
        
        return self._get_obs(), reward, self.done, self.truncated, info
    
    def render(self):
        """Render the current game state."""
        if self.render_mode != "human":
            return
        
        print(f"\n{'='*50}")
        print(f"  MULTIPLAYER UNO - {self.num_players} Players")
        print(f"{'='*50}")
        print(f"Open Card: [{self.card_open.color} {self.card_open.value}]")
        print(f"Direction: {'→ Clockwise' if self.direction == 1 else '← Counter-clockwise'}")
        print(f"Deck: {len(self.deck.cards)} cards")
        print()
        
        for i in range(self.num_players):
            marker = "★ YOU" if i == 0 else f"  P{i+1}"
            cards = len(self.hands[i])
            if i == 0:
                hand_str = " ".join(f"[{c.color} {c.value}]" for c in self.hands[i])
                print(f"{marker} ({cards} cards): {hand_str}")
            else:
                print(f"{marker} ({cards} cards)")
        
        print(f"{'='*50}")
    
    def get_win_rate(self) -> float:
        """Get current win rate."""
        return self.wins / self.games_played if self.games_played > 0 else 0.0


# Wrapper for 3-player games
class ThreePlayerUnoEnv(MultiplayerUnoEnv):
    """3-player UNO environment."""
    def __init__(self, **kwargs):
        super().__init__(num_players=3, **kwargs)


# Wrapper for 4-player games  
class FourPlayerUnoEnv(MultiplayerUnoEnv):
    """4-player UNO environment."""
    def __init__(self, **kwargs):
        super().__init__(num_players=4, **kwargs)
