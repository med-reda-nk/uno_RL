"""
Training script for RL agents in UNO game.
Supports DQN and Improved Q-Learning agents.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn_agent import DQNAgent, ImprovedQLearningAgent
from src.players import Player
from src.turn import Turn
from src.cards import Deck
from src.utils import check_win, block_print, enable_print
import config as conf


class TrainingGame:
    """
    A game instance for training RL agents.
    Optimized for fast training with minimal output.
    """
    
    def __init__(self, agent, opponent_type="random", verbose=False):
        """
        Initialize a training game.
        
        Parameters:
        -----------
        agent : DQNAgent or ImprovedQLearningAgent
            The agent being trained
        opponent_type : str
            Type of opponent: "random" or "self"
        verbose : bool
            Whether to print game details
        """
        self.agent = agent
        self.opponent_type = opponent_type
        self.verbose = verbose
        
        if not verbose:
            block_print()
    
    def play(self):
        """
        Play a single game and return the result.
        
        Returns:
        --------
        tuple: (winner_name, num_turns, agent_rewards)
        """
        # Initialize game
        deck = Deck()
        player_1 = Player("Agent", self.agent)
        player_2 = Player("Opponent", self.agent)
        
        # Deal initial cards
        card_open = deck.draw_from_deck()
        while card_open.value not in range(0, 10):
            card_open = deck.draw_from_deck()
        
        for _ in range(7):
            self._draw_card(player_1, deck, card_open)
            self._draw_card(player_2, deck, card_open)
        
        turn = 0
        max_turns = 500
        current_player = player_1
        other_player = player_2
        
        while turn < max_turns:
            turn += 1
            
            # Evaluate playable cards
            current_player.evaluate_hand(card_open)
            
            # Draw if no playable cards
            if len(current_player.hand_play) == 0:
                self._draw_card(current_player, deck, card_open)
                current_player.evaluate_hand(card_open)
            
            # Play a card if possible
            if len(current_player.hand_play) > 0:
                if current_player.name == "Agent":
                    card_open = self._agent_play(current_player, deck, card_open)
                else:
                    if self.opponent_type == "self":
                        card_open = self._agent_play(current_player, deck, card_open)
                    else:
                        card_open = self._random_play(current_player, deck)
                
                # Handle special cards
                if card_open.value == "PL2":
                    for _ in range(2):
                        self._draw_card(other_player, deck, card_open)
                elif card_open.value == "PL4":
                    for _ in range(4):
                        self._draw_card(other_player, deck, card_open)
                elif card_open.value in ["SKI", "REV"]:
                    # Skip/reverse gives extra turn
                    continue
            
            # Check for win
            if len(current_player.hand) == 0:
                winner = current_player.name
                break
            
            # Switch players
            current_player, other_player = other_player, current_player
        else:
            # Game went too long, determine winner by hand size
            if len(player_1.hand) < len(player_2.hand):
                winner = "Agent"
            elif len(player_2.hand) < len(player_1.hand):
                winner = "Opponent"
            else:
                winner = "Draw"
        
        if not self.verbose:
            enable_print()
        
        return winner, turn
    
    def _draw_card(self, player, deck, card_open):
        """Draw a card for a player."""
        if len(deck.cards) == 0:
            deck.cards = deck.cards_disc
            deck.cards_disc = []
            deck.shuffle()
        
        card = deck.draw_from_deck()
        player.hand.append(card)
        player.evaluate_hand(card_open)
    
    def _agent_play(self, player, deck, card_open):
        """Agent plays using the RL policy."""
        player.identify_state(card_open)
        player.identify_action()
        
        action = self.agent.step(player.state, player.actions)
        
        # Find and play the card
        card_played = self._find_card_for_action(player, action, card_open)
        
        if card_played:
            player.hand.remove(card_played)
            deck.discard(card_played)
            
            # Handle wild cards
            if card_played.value in ["COL", "PL4"]:
                colors = [c.color for c in player.hand if c.color in ["RED", "GRE", "BLU", "YEL"]]
                card_played.color = max(set(colors), key=colors.count) if colors else "RED"
            
            # Update agent
            player.identify_state(card_played)
            self.agent.update(player.state, action)
            
            return card_played
        
        return card_open
    
    def _random_play(self, player, deck):
        """Opponent plays randomly."""
        import random
        if player.hand_play:
            card = random.choice(player.hand_play)
            player.hand.remove(card)
            player.hand_play.remove(card)
            deck.discard(card)
            
            if card.value in ["COL", "PL4"]:
                card.color = random.choice(["RED", "GRE", "BLU", "YEL"])
            
            return card
        return None
    
    def _find_card_for_action(self, player, action, card_open):
        """Find a card in hand that matches the action."""
        for card in player.hand_play:
            if action in ["COL", "PL4"] and card.value == action:
                return card
            if action in ["RED", "GRE", "BLU", "YEL"]:
                if card.color == action and card.value in range(0, 10):
                    return card
            if action in ["SKI", "REV", "PL2"] and card.value == action:
                return card
        
        # Fallback: return any playable card
        return player.hand_play[0] if player.hand_play else None


def train_agent(
    agent_type="dqn",
    num_episodes=1000,
    opponent_type="random",
    save_freq=100,
    verbose=False,
    config=None
):
    """
    Train an RL agent to play UNO.
    
    Parameters:
    -----------
    agent_type : str
        Type of agent: "dqn" or "qlearning"
    num_episodes : int
        Number of games to train on
    opponent_type : str
        Type of opponent: "random" or "self"
    save_freq : int
        How often to save the model
    verbose : bool
        Whether to print detailed output
    config : dict
        Agent configuration parameters
    """
    print(f"\n{'='*60}")
    print(f"Training {agent_type.upper()} Agent on UNO")
    print(f"{'='*60}")
    
    # Default configuration
    if config is None:
        if agent_type == "dqn":
            config = {
                "epsilon": 1.0,
                "epsilon_decay": 0.998,
                "epsilon_min": 0.01,
                "gamma": 0.99,
                "learning_rate": 0.001,
                "batch_size": 64,
                "buffer_size": 20000,
                "target_update_freq": 100,
                "hidden_sizes": [128, 128, 64]
            }
        else:
            config = {
                "epsilon": 0.5,
                "epsilon_decay": 0.999,
                "epsilon_min": 0.05,
                "step_size": 0.1,
                "gamma": 0.95,
                "batch_size": 32,
                "buffer_size": 10000
            }
    
    # Create agent
    if agent_type == "dqn":
        agent = DQNAgent(config)
        model_path = "models/dqn_agent.pkl"
    else:
        agent = ImprovedQLearningAgent(config)
        model_path = "models/qlearning_agent.pkl"
    
    # Try to load existing model
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        agent.load(model_path)
    
    # Training metrics
    wins = 0
    losses = 0
    draws = 0
    win_rates = []
    avg_turns = []
    recent_wins = deque(maxlen=100)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    print(f"\nTraining for {num_episodes} episodes...")
    print(f"Opponent type: {opponent_type}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    print(f"\n{'='*60}\n")
    
    start_time = time.time()
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        game = TrainingGame(agent, opponent_type, verbose)
        winner, turns = game.play()
        
        # Update statistics
        if winner == "Agent":
            wins += 1
            recent_wins.append(1)
            agent.end_episode(won=True)
        elif winner == "Opponent":
            losses += 1
            recent_wins.append(0)
            agent.end_episode(won=False)
        else:
            draws += 1
            recent_wins.append(0.5)
            agent.end_episode(won=False)
        
        win_rates.append(sum(recent_wins) / len(recent_wins))
        avg_turns.append(turns)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            recent_win_rate = sum(recent_wins) / len(recent_wins)
            stats = agent.get_training_stats()
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Win Rate (last 100): {recent_win_rate:.2%}")
            print(f"  Overall: {wins}W / {losses}L / {draws}D")
            print(f"  Epsilon: {stats['epsilon']:.4f}")
            print(f"  Avg Turns: {np.mean(avg_turns[-100:]):.1f}")
            if 'avg_loss' in stats:
                print(f"  Avg Loss: {stats.get('avg_loss', 0):.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(model_path)
    
    # Final save
    agent.save(model_path)
    
    elapsed_time = time.time() - start_time
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"\nTotal Episodes: {num_episodes}")
    print(f"Time Elapsed: {elapsed_time/60:.2f} minutes")
    print(f"Games/Second: {num_episodes/elapsed_time:.2f}")
    print(f"\nFinal Statistics:")
    print(f"  Wins: {wins} ({wins/num_episodes:.2%})")
    print(f"  Losses: {losses} ({losses/num_episodes:.2%})")
    print(f"  Draws: {draws} ({draws/num_episodes:.2%})")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")
    
    # Save training curves
    save_training_plots(win_rates, avg_turns, agent_type)
    
    # Save training results to CSV
    results_df = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'win_rate': win_rates,
        'turns': avg_turns
    })
    results_df.to_csv(f"assets/{agent_type}_training_results.csv", index=False)
    
    return agent


def save_training_plots(win_rates, avg_turns, agent_type):
    """Save training progress plots."""
    os.makedirs("assets", exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Win rate plot
    axes[0].plot(win_rates, alpha=0.3, color='blue')
    # Smoothed win rate
    window = min(100, len(win_rates))
    if window > 0:
        smoothed = pd.Series(win_rates).rolling(window=window).mean()
        axes[0].plot(smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Win Rate')
    axes[0].set_title(f'{agent_type.upper()} Agent - Win Rate Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Average turns plot
    axes[1].plot(avg_turns, alpha=0.3, color='green')
    if window > 0:
        smoothed_turns = pd.Series(avg_turns).rolling(window=window).mean()
        axes[1].plot(smoothed_turns, color='green', linewidth=2, label=f'Smoothed (window={window})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Number of Turns')
    axes[1].set_title('Average Game Length Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"assets/{agent_type}_training_curves.png", dpi=150)
    plt.close()
    
    print(f"\nTraining plots saved to assets/{agent_type}_training_curves.png")


def evaluate_agent(agent, num_games=100, opponent_type="random"):
    """
    Evaluate a trained agent.
    
    Parameters:
    -----------
    agent : DQNAgent or ImprovedQLearningAgent
        The trained agent
    num_games : int
        Number of games to evaluate
    opponent_type : str
        Type of opponent
    
    Returns:
    --------
    dict: Evaluation metrics
    """
    print(f"\n{'='*60}")
    print("Evaluating Agent")
    print(f"{'='*60}")
    
    # Set epsilon to 0 for evaluation (greedy policy)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    wins = 0
    total_turns = 0
    
    for _ in tqdm(range(num_games), desc="Evaluating"):
        game = TrainingGame(agent, opponent_type, verbose=False)
        winner, turns = game.play()
        
        if winner == "Agent":
            wins += 1
        total_turns += turns
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    metrics = {
        "games": num_games,
        "wins": wins,
        "win_rate": wins / num_games,
        "avg_turns": total_turns / num_games
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Games Played: {num_games}")
    print(f"  Wins: {wins}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Avg Turns: {metrics['avg_turns']:.1f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for UNO")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "qlearning"],
                        help="Type of agent to train")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train")
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "self"],
                        help="Type of opponent")
    parser.add_argument("--save-freq", type=int, default=100,
                        help="How often to save the model")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed game output")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate after training")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Number of games for evaluation")
    
    args = parser.parse_args()
    
    # Train the agent
    agent = train_agent(
        agent_type=args.agent,
        num_episodes=args.episodes,
        opponent_type=args.opponent,
        save_freq=args.save_freq,
        verbose=args.verbose
    )
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_agent(agent, num_games=args.eval_games, opponent_type=args.opponent)


if __name__ == "__main__":
    main()
