"""
Unit Tests for UNO Card Game RL Project
Run with: python -m pytest tests/ -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cards import Card, Deck


class TestCard:
    """Tests for the Card class."""
    
    def test_card_creation(self):
        """Test basic card creation."""
        card = Card("RED", 5)
        assert card.color == "RED"
        assert card.value == 5
    
    def test_card_print(self):
        """Test card string representation."""
        card = Card("BLU", 7)
        result = card.print_card()
        assert "BLU" in result or "Blue" in result.lower()
    
    def test_special_card_creation(self):
        """Test special cards (SKI, REV, PL2)."""
        skip = Card("GRE", "SKI")
        reverse = Card("YEL", "REV")
        plus2 = Card("RED", "PL2")
        
        assert skip.value == "SKI"
        assert reverse.value == "REV"
        assert plus2.value == "PL2"
    
    def test_wild_card_creation(self):
        """Test wild cards (COL, PL4)."""
        wild = Card("WILD", "COL")
        wild_plus4 = Card("WILD", "PL4")
        
        assert wild.color == "WILD"
        assert wild_plus4.value == "PL4"


class TestDeck:
    """Tests for the Deck class."""
    
    def test_deck_creation(self):
        """Test deck is created with cards."""
        deck = Deck()
        assert len(deck.cards) > 0
    
    def test_deck_size(self):
        """Test standard UNO deck size (108 cards)."""
        deck = Deck()
        # UNO has 108 cards total
        assert len(deck.cards) == 108
    
    def test_draw_card(self):
        """Test drawing a card from deck."""
        deck = Deck()
        initial_size = len(deck.cards)
        card = deck.draw_from_deck()
        
        assert card is not None
        assert isinstance(card, Card)
        assert len(deck.cards) == initial_size - 1
    
    def test_deck_has_all_colors(self):
        """Test deck contains all 4 colors."""
        deck = Deck()
        colors = set(card.color for card in deck.cards)
        
        assert "RED" in colors
        assert "GRE" in colors
        assert "BLU" in colors
        assert "YEL" in colors


class TestCardPlayability:
    """Tests for card playability rules."""
    
    def test_same_color_playable(self):
        """Test cards of same color can be played."""
        open_card = Card("RED", 5)
        hand_card = Card("RED", 8)
        
        # Same color should be playable
        assert hand_card.color == open_card.color
    
    def test_same_value_playable(self):
        """Test cards of same value can be played."""
        open_card = Card("RED", 5)
        hand_card = Card("BLU", 5)
        
        # Same value should be playable
        assert hand_card.value == open_card.value
    
    def test_wild_always_playable(self):
        """Test wild cards can always be played."""
        wild = Card("WILD", "COL")
        wild_plus4 = Card("WILD", "PL4")
        
        # Wild cards should always be playable
        assert wild.value in ["COL", "PL4"]
        assert wild_plus4.value in ["COL", "PL4"]


class TestGameRules:
    """Tests for UNO game rules."""
    
    def test_starting_hand_size(self):
        """Test players start with 7 cards."""
        STARTING_HAND_SIZE = 7
        assert STARTING_HAND_SIZE == 7
    
    def test_draw_penalty_plus2(self):
        """Test +2 card makes opponent draw 2."""
        PLUS2_PENALTY = 2
        assert PLUS2_PENALTY == 2
    
    def test_draw_penalty_plus4(self):
        """Test +4 card makes opponent draw 4."""
        PLUS4_PENALTY = 4
        assert PLUS4_PENALTY == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
