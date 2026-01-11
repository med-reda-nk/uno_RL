"""
UNO Card and Deck classes for the card game.

This module provides the core data structures for representing
UNO cards and the deck they come from.
"""

import random
from typing import List, Union, Optional


# Type alias for card values
CardValue = Union[int, str]  # 0-9 or "SKI", "REV", "PL2", "COL", "PL4"
CardColor = str  # "RED", "GRE", "BLU", "YEL", "WILD"


class Card:
    """
    Represents a single UNO card with color and value.
    
    Attributes:
        color: Card color - "RED", "GRE", "BLU", "YEL", or "WILD"
        value: Card value - 0-9 for number cards, or special values:
               "SKI" (Skip), "REV" (Reverse), "PL2" (+2),
               "COL" (Wild Color), "PL4" (Wild +4)
    
    Example:
        >>> card = Card("RED", 5)
        >>> card.evaluate_card("RED", 3)  # Same color
        True
        >>> card.evaluate_card("BLU", 5)  # Same value
        True
    """
    
    def __init__(self, color: CardColor, value: CardValue) -> None:
        """
        Initialize a card with color and value.
        
        Args:
            color: The card's color ("RED", "GRE", "BLU", "YEL", "WILD")
            value: The card's value (0-9 or special string)
        """
        self.color = color
        self.value = value
    
    def evaluate_card(self, open_color: CardColor, open_value: CardValue) -> bool:
        """
        Check if this card can be played on the given open card.
        
        A card is playable if:
        - It matches the open card's color, OR
        - It matches the open card's value, OR
        - It's a wild card (COL or PL4)
        
        Args:
            open_color: The color of the card currently in play
            open_value: The value of the card currently in play
            
        Returns:
            True if this card can be played, False otherwise
        """
        if (
            (self.color == open_color) or 
            (self.value == open_value) or 
            (self.value in ["COL", "PL4"])
        ):
            return True
        return False
    
    def show_card(self) -> None:
        """Print the card's color and value to console."""
        print(self.color, self.value)
    
    def print_card(self) -> str:
        """
        Get a string representation of the card.
        
        Returns:
            String in format "COLOR VALUE" (e.g., "RED 5")
        """
        return f"{self.color} {self.value}"
    
    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return f"Card({self.color!r}, {self.value!r})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another card."""
        if not isinstance(other, Card):
            return NotImplemented
        return self.color == other.color and self.value == other.value


class Deck:
    """
    Represents a standard UNO deck of 108 cards.
    
    A standard UNO deck contains:
    - 76 Number cards (0-9 in each of 4 colors, with one 0 and two of 1-9)
    - 24 Action cards (2 each of Skip, Reverse, +2 in each of 4 colors)
    - 8 Wild cards (4 Wild and 4 Wild +4)
    
    Attributes:
        cards: List of cards currently in the draw pile
        cards_disc: List of discarded cards
        
    Example:
        >>> deck = Deck()
        >>> len(deck.cards)
        108
        >>> card = deck.draw_from_deck()
        >>> deck.discard(card)
    """
    
    # UNO card colors
    COLORS: List[str] = ["RED", "GRE", "BLU", "YEL"]
    
    # Special card values
    ACTION_VALUES: List[str] = ["SKI", "REV", "PL2"]
    WILD_VALUES: List[str] = ["COL", "PL4"]
    
    def __init__(self) -> None:
        """Initialize a new shuffled deck of 108 UNO cards."""
        self.cards: List[Card] = []
        self.cards_disc: List[Card] = []
        self.build()
        self.shuffle()
    
    def build(self) -> None:
        """
        Build the standard 108-card UNO deck.
        
        Creates:
        - One 0 card per color (4 total)
        - Two of each 1-9 per color (72 total)
        - Two of each action card per color (24 total)
        - Four of each wild card (8 total)
        """
        # Zero cards - one per color
        cards_zero = [Card(c, 0) for c in self.COLORS]
        
        # Number cards 1-9 - two per color
        cards_normal = [Card(c, v) for c in self.COLORS for v in range(1, 10)] * 2
        
        # Action cards - two per color
        cards_action = [Card(c, v) for c in self.COLORS for v in self.ACTION_VALUES] * 2
        
        # Wild cards - four of each
        cards_wild = [Card("WILD", v) for v in self.WILD_VALUES] * 4
        
        # Combine all cards
        cards_all = cards_normal + cards_action + cards_zero + cards_wild
        self.cards.extend(cards_all)
    
    def discard(self, card: Card) -> None:
        """
        Add a card to the discard pile.
        
        Args:
            card: The card to discard
        """
        self.cards_disc.append(card)
    
    def shuffle(self) -> None:
        """Randomly shuffle the cards in the draw pile."""
        random.shuffle(self.cards)

    def draw_from_deck(self) -> Card:
        """
        Draw the top card from the deck.
        
        If the draw pile is empty, the discard pile is shuffled
        and becomes the new draw pile.
        
        Returns:
            The drawn card
            
        Raises:
            IndexError: If both draw and discard piles are empty
        """
        if len(self.cards) == 0:
            # Reshuffle discard pile into draw pile
            self.cards = self.cards_disc
            self.cards_disc = []
            self.shuffle()
            
        return self.cards.pop()
    
    def show_deck(self) -> None:
        """Print all cards in the draw pile."""
        for card in self.cards:
            card.show_card()
    
    def show_discarded(self) -> None:
        """Print all cards in the discard pile."""
        for card in self.cards_disc:
            card.show_card()
    
    def __len__(self) -> int:
        """Return the number of cards in the draw pile."""
        return len(self.cards)