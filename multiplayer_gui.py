"""
Multiplayer UNO GUI (3-4 Players)
=================================
Play UNO with multiple AI opponents!

Features:
- 3 or 4 player modes
- Multiple AI opponents with different strategies
- Visual hand display for all players
- Turn indicator and direction arrows
"""

import pygame
import sys
import random
import math
import os
from typing import List, Optional, Tuple, Dict

# Import base game components
try:
    import numpy as np
    from src.cards import Deck, Card
    GAME_AVAILABLE = True
except ImportError:
    GAME_AVAILABLE = False
    print("Error: Game components not found.")
    sys.exit(1)

# Try to import trained models
try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: Models not available. AI will use random strategy.")

# Initialize Pygame
pygame.init()
pygame.font.init()

# =============================================================================
# CONSTANTS
# =============================================================================

WINDOW_WIDTH = 1050
WINDOW_HEIGHT = 700
FPS = 60

COLORS = {
    'RED': (239, 68, 68),
    'GRE': (16, 185, 129),
    'BLU': (59, 130, 246),
    'YEL': (250, 204, 21),
    'WILD': (50, 50, 70),
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'BG_TOP': (5, 8, 22),
    'BG_BOTTOM': (15, 23, 42),
    'PANEL_BG': (20, 30, 50),
    'ACCENT_PURPLE': (192, 132, 252),
    'ACCENT_CYAN': (34, 211, 238),
    'ACCENT_ORANGE': (251, 146, 60),
    'ACCENT_PINK': (244, 114, 182),
    'GOLD': (251, 191, 36),
    'GRAY': (100, 100, 120),
}

CARD_WIDTH = 55
CARD_HEIGHT = 75
CARD_RADIUS = 6

# Fonts
try:
    FONT_LARGE = pygame.font.SysFont('segoeui', 32, bold=True)
    FONT_MEDIUM = pygame.font.SysFont('segoeui', 20, bold=True)
    FONT_REGULAR = pygame.font.SysFont('segoeui', 16)
    FONT_SMALL = pygame.font.SysFont('segoeui', 12)
    FONT_CARD = pygame.font.SysFont('segoeui', 16, bold=True)
except:
    FONT_LARGE = pygame.font.Font(None, 52)
    FONT_MEDIUM = pygame.font.Font(None, 32)
    FONT_REGULAR = pygame.font.Font(None, 24)
    FONT_SMALL = pygame.font.Font(None, 20)
    FONT_CARD = pygame.font.Font(None, 28)

# Player positions (for 4 players)
PLAYER_POSITIONS = {
    0: {"name": "You", "pos": "bottom", "color": COLORS['ACCENT_CYAN']},
    1: {"name": "AI 1", "pos": "left", "color": COLORS['ACCENT_ORANGE']},
    2: {"name": "AI 2", "pos": "top", "color": COLORS['ACCENT_PURPLE']},
    3: {"name": "AI 3", "pos": "right", "color": COLORS['ACCENT_PINK']},
}


# =============================================================================
# CARD CLASS
# =============================================================================

class GUICard:
    """Card with rendering capabilities."""
    
    def __init__(self, color: str, value):
        self.color = color
        self.value = value
        self.rect = pygame.Rect(0, 0, CARD_WIDTH, CARD_HEIGHT)
        self.hover = False
    
    def is_playable(self, open_card) -> bool:
        """Check if this card can be played."""
        return (
            self.color == open_card.color or
            self.value == open_card.value or
            self.value in ["COL", "PL4"]
        )
    
    def get_display_value(self) -> str:
        if self.value == "SKI":
            return "O/"
        elif self.value == "REV":
            return "<>"
        elif self.value == "PL2":
            return "+2"
        elif self.value == "PL4":
            return "+4"
        elif self.value == "COL":
            return "W"
        else:
            return str(self.value)
    
    def get_color_rgb(self) -> Tuple[int, int, int]:
        return COLORS.get(self.color, COLORS['WILD'])
    
    def render(self, surface: pygame.Surface, x: int, y: int, 
               face_up: bool = True, highlighted: bool = False):
        """Render the card."""
        self.rect.x = x
        self.rect.y = y
        
        if face_up:
            # Card background
            color = self.get_color_rgb()
            pygame.draw.rect(surface, color, self.rect, border_radius=CARD_RADIUS)
            
            # Border
            border_color = COLORS['WHITE'] if highlighted or self.hover else (50, 50, 50)
            border_width = 3 if highlighted or self.hover else 2
            pygame.draw.rect(surface, border_color, self.rect, border_width, border_radius=CARD_RADIUS)
            
            # Value
            value_text = self.get_display_value()
            text_color = COLORS['BLACK'] if self.color == 'YEL' else COLORS['WHITE']
            text_surf = FONT_CARD.render(value_text, True, text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        else:
            # Card back
            pygame.draw.rect(surface, COLORS['WILD'], self.rect, border_radius=CARD_RADIUS)
            pygame.draw.rect(surface, COLORS['GRAY'], self.rect, 2, border_radius=CARD_RADIUS)
            
            # UNO text
            text_surf = FONT_SMALL.render("UNO", True, COLORS['RED'])
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)


# =============================================================================
# GAME CLASS
# =============================================================================

class MultiplayerUnoGame:
    """Multiplayer UNO Game with GUI."""
    
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(f"UNO - {num_players} Players")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.deck: Optional[Deck] = None
        self.hands: List[List[GUICard]] = []
        self.card_open: Optional[GUICard] = None
        self.current_player: int = 0
        self.direction: int = 1  # 1 = clockwise
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.message: str = ""
        self.message_timer: int = 0
        
        # Animation state
        self.animation_card: Optional[GUICard] = None
        self.animation_progress: float = 0
        self.animation_start: Tuple[int, int] = (0, 0)
        self.animation_end: Tuple[int, int] = (0, 0)
        
        # Color selection for wild cards
        self.selecting_color: bool = False
        self.pending_wild_card: Optional[GUICard] = None
        
        # AI delay for visibility
        self.ai_delay: int = 0
        self.ai_delay_max: int = 45  # frames
        
        # Load AI model if available
        self.ai_model = None
        self._load_ai_model()
        
        self.new_game()
    
    def _load_ai_model(self):
        """Load trained AI model for opponents."""
        if not MODELS_AVAILABLE:
            return
        
        model_paths = [
            "models/selfplay_champion.zip",
            "models/best_recurrent_ppo_uno.zip",
            "models/sb3_ppo_uno.zip",
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    if "recurrent" in path.lower():
                        self.ai_model = RecurrentPPO.load(path)
                    else:
                        self.ai_model = PPO.load(path)
                    print(f"Loaded AI model: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
    
    def new_game(self):
        """Start a new game."""
        # Initialize deck
        self.deck = Deck()
        
        # Deal hands
        self.hands = [[] for _ in range(self.num_players)]
        for _ in range(7):
            for player_idx in range(self.num_players):
                card = self.deck.draw_from_deck()
                self.hands[player_idx].append(GUICard(card.color, card.value))
        
        # Draw open card (must be number)
        card = self.deck.draw_from_deck()
        while not isinstance(card.value, int):
            self.deck.discard(card)
            card = self.deck.draw_from_deck()
        self.card_open = GUICard(card.color, card.value)
        
        # Reset state
        self.current_player = 0
        self.direction = 1
        self.game_over = False
        self.winner = None
        self.selecting_color = False
        self.pending_wild_card = None
        self.ai_delay = 0
        
        self.show_message("Game started! Your turn.")
    
    def show_message(self, text: str, duration: int = 120):
        """Show a temporary message."""
        self.message = text
        self.message_timer = duration
    
    def get_playable_cards(self, player_idx: int) -> List[int]:
        """Get indices of playable cards for a player."""
        playable = []
        for i, card in enumerate(self.hands[player_idx]):
            if card.is_playable(self.card_open):
                playable.append(i)
        return playable
    
    def play_card(self, player_idx: int, card_idx: int, chosen_color: Optional[str] = None):
        """Play a card from a player's hand."""
        card = self.hands[player_idx][card_idx]
        
        # Handle wild cards
        if card.value in ["COL", "PL4"]:
            if chosen_color:
                card.color = chosen_color
            else:
                # AI picks random color (or most common in hand)
                colors = [c.color for c in self.hands[player_idx] 
                         if c.color in ["RED", "GRE", "BLU", "YEL"]]
                card.color = max(set(colors), key=colors.count) if colors else random.choice(["RED", "GRE", "BLU", "YEL"])
        
        # Remove from hand
        self.hands[player_idx].pop(card_idx)
        
        # Apply card effects
        next_player = (player_idx + self.direction) % self.num_players
        
        if card.value == "SKI":
            # Skip next player
            self.show_message(f"{PLAYER_POSITIONS[next_player]['name']} skipped!")
            self.current_player = (next_player + self.direction) % self.num_players
        elif card.value == "REV":
            # Reverse direction
            self.direction *= -1
            self.show_message("Direction reversed!")
            if self.num_players == 2:
                self.current_player = player_idx
            else:
                self.current_player = (player_idx + self.direction) % self.num_players
        elif card.value == "PL2":
            # Draw 2
            for _ in range(2):
                if self.deck.cards:
                    c = self.deck.draw_from_deck()
                    self.hands[next_player].append(GUICard(c.color, c.value))
            self.show_message(f"{PLAYER_POSITIONS[next_player]['name']} draws 2!")
            self.current_player = (next_player + self.direction) % self.num_players
        elif card.value == "PL4":
            # Draw 4
            for _ in range(4):
                if self.deck.cards:
                    c = self.deck.draw_from_deck()
                    self.hands[next_player].append(GUICard(c.color, c.value))
            self.show_message(f"{PLAYER_POSITIONS[next_player]['name']} draws 4!")
            self.current_player = (next_player + self.direction) % self.num_players
        else:
            self.current_player = next_player
        
        # Set as open card
        self.card_open = card
        
        # Check for win
        if len(self.hands[player_idx]) == 0:
            self.game_over = True
            self.winner = player_idx
            if player_idx == 0:
                self.show_message("YOU WIN!", 300)
            else:
                self.show_message(f"{PLAYER_POSITIONS[player_idx]['name']} wins!", 300)
    
    def draw_card(self, player_idx: int):
        """Draw a card for a player."""
        if self.deck.cards:
            card = self.deck.draw_from_deck()
            self.hands[player_idx].append(GUICard(card.color, card.value))
    
    def ai_turn(self):
        """Execute AI player's turn."""
        if self.ai_delay > 0:
            self.ai_delay -= 1
            return
        
        player_idx = self.current_player
        playable = self.get_playable_cards(player_idx)
        
        if playable:
            # Choose a card to play
            # Prioritize special cards
            best_idx = playable[0]
            for idx in playable:
                card = self.hands[player_idx][idx]
                if card.value in ["SKI", "REV", "PL2", "PL4"]:
                    best_idx = idx
                    break
            
            self.play_card(player_idx, best_idx)
        else:
            # Draw a card
            self.draw_card(player_idx)
            self.current_player = (player_idx + self.direction) % self.num_players
        
        self.ai_delay = self.ai_delay_max
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click."""
        # Check close button (top-right corner)
        close_rect = pygame.Rect(WINDOW_WIDTH - 45, 10, 35, 35)
        if close_rect.collidepoint(pos):
            pygame.quit()
            sys.exit()
        
        if self.game_over:
            # Check for new game button
            button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 50, 200, 50)
            if button_rect.collidepoint(pos):
                self.new_game()
            return
        
        if self.selecting_color:
            # Check color buttons
            colors = ["RED", "GRE", "BLU", "YEL"]
            for i, color in enumerate(colors):
                x = WINDOW_WIDTH // 2 - 150 + i * 80
                y = WINDOW_HEIGHT // 2
                rect = pygame.Rect(x, y, 60, 60)
                if rect.collidepoint(pos):
                    self.play_card(0, self.hands[0].index(self.pending_wild_card), color)
                    self.selecting_color = False
                    self.pending_wild_card = None
                    return
            return
        
        if self.current_player != 0:
            return  # Not player's turn
        
        # Check card clicks
        playable = self.get_playable_cards(0)
        hand = self.hands[0]
        
        # Calculate card positions (same as rendering)
        total_width = len(hand) * 50 + CARD_WIDTH
        start_x = (WINDOW_WIDTH - total_width) // 2
        y = WINDOW_HEIGHT - CARD_HEIGHT - 80
        
        for i, card in enumerate(hand):
            x = start_x + i * 50
            rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
            if rect.collidepoint(pos):
                if i in playable:
                    if card.value in ["COL", "PL4"]:
                        self.selecting_color = True
                        self.pending_wild_card = card
                    else:
                        self.play_card(0, i)
                else:
                    self.show_message("Can't play that card!", 60)
                return
        
        # Check draw pile click
        deck_rect = pygame.Rect(WINDOW_WIDTH // 2 + 80, WINDOW_HEIGHT // 2 - 50, CARD_WIDTH, CARD_HEIGHT)
        if deck_rect.collidepoint(pos) and not playable:
            self.draw_card(0)
            self.current_player = (0 + self.direction) % self.num_players
            self.ai_delay = self.ai_delay_max
    
    def update(self):
        """Update game state."""
        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= 1
        
        # AI turns
        if not self.game_over and self.current_player != 0:
            self.ai_turn()
    
    def render_background(self):
        """Render gradient background."""
        for y in range(WINDOW_HEIGHT):
            ratio = y / WINDOW_HEIGHT
            r = int(COLORS['BG_TOP'][0] * (1 - ratio) + COLORS['BG_BOTTOM'][0] * ratio)
            g = int(COLORS['BG_TOP'][1] * (1 - ratio) + COLORS['BG_BOTTOM'][1] * ratio)
            b = int(COLORS['BG_TOP'][2] * (1 - ratio) + COLORS['BG_BOTTOM'][2] * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))
    
    def render_player_hand(self, player_idx: int):
        """Render a player's hand."""
        hand = self.hands[player_idx]
        is_current = (self.current_player == player_idx)
        info = PLAYER_POSITIONS[player_idx]
        
        if info["pos"] == "bottom":
            # Player's hand (face up, at bottom)
            total_width = len(hand) * 50 + CARD_WIDTH
            start_x = (WINDOW_WIDTH - total_width) // 2
            y = WINDOW_HEIGHT - CARD_HEIGHT - 80
            
            playable = self.get_playable_cards(0) if player_idx == 0 and self.current_player == 0 else []
            
            for i, card in enumerate(hand):
                highlighted = i in playable
                card.render(self.screen, start_x + i * 50, y, True, highlighted)
            
            # Name label
            label = FONT_MEDIUM.render(f"{info['name']} ({len(hand)} cards)", True, info['color'])
            self.screen.blit(label, (WINDOW_WIDTH // 2 - label.get_width() // 2, WINDOW_HEIGHT - 40))
            
        elif info["pos"] == "top":
            # Top player (face down)
            total_width = min(len(hand) * 30 + CARD_WIDTH, 400)
            start_x = (WINDOW_WIDTH - total_width) // 2
            y = 60
            
            spacing = min(30, (total_width - CARD_WIDTH) // max(len(hand) - 1, 1)) if len(hand) > 1 else 0
            for i, card in enumerate(hand):
                card.render(self.screen, start_x + i * spacing, y, False)
            
            # Name label
            label = FONT_MEDIUM.render(f"{info['name']} ({len(hand)})", True, info['color'])
            self.screen.blit(label, (WINDOW_WIDTH // 2 - label.get_width() // 2, 20))
            
        elif info["pos"] == "left":
            # Left player (face down, vertical)
            start_y = (WINDOW_HEIGHT - min(len(hand) * 25 + CARD_HEIGHT, 350)) // 2
            x = 60
            
            spacing = min(25, (350 - CARD_HEIGHT) // max(len(hand) - 1, 1)) if len(hand) > 1 else 0
            for i, card in enumerate(hand):
                card.render(self.screen, x, start_y + i * spacing, False)
            
            # Name label
            label = FONT_MEDIUM.render(f"{info['name']} ({len(hand)})", True, info['color'])
            self.screen.blit(label, (20, start_y - 40))
            
        elif info["pos"] == "right":
            # Right player (face down, vertical)
            start_y = (WINDOW_HEIGHT - min(len(hand) * 25 + CARD_HEIGHT, 350)) // 2
            x = WINDOW_WIDTH - CARD_WIDTH - 60
            
            spacing = min(25, (350 - CARD_HEIGHT) // max(len(hand) - 1, 1)) if len(hand) > 1 else 0
            for i, card in enumerate(hand):
                card.render(self.screen, x, start_y + i * spacing, False)
            
            # Name label
            label = FONT_MEDIUM.render(f"{info['name']} ({len(hand)})", True, info['color'])
            self.screen.blit(label, (WINDOW_WIDTH - 20 - label.get_width(), start_y - 40))
        
        # Current player indicator
        if is_current and not self.game_over:
            indicator_text = ">> TURN <<" if player_idx == 0 else "Thinking..."
            indicator = FONT_SMALL.render(indicator_text, True, COLORS['GOLD'])
            
            if info["pos"] == "bottom":
                self.screen.blit(indicator, (WINDOW_WIDTH // 2 - indicator.get_width() // 2, WINDOW_HEIGHT - 65))
            elif info["pos"] == "top":
                self.screen.blit(indicator, (WINDOW_WIDTH // 2 - indicator.get_width() // 2, 165))
            elif info["pos"] == "left":
                self.screen.blit(indicator, (60, WINDOW_HEIGHT // 2 + 180))
            elif info["pos"] == "right":
                self.screen.blit(indicator, (WINDOW_WIDTH - 140, WINDOW_HEIGHT // 2 + 180))
    
    def render_center(self):
        """Render center area (open card, deck, direction)."""
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        
        # Draw pile
        deck_card = GUICard("WILD", "")
        deck_card.render(self.screen, center_x + 60, center_y - CARD_HEIGHT // 2, False)
        
        # Deck count
        deck_text = FONT_SMALL.render(f"{len(self.deck.cards)}", True, COLORS['WHITE'])
        self.screen.blit(deck_text, (center_x + 80, center_y + CARD_HEIGHT // 2 + 10))
        
        # Open card
        if self.card_open:
            self.card_open.render(self.screen, center_x - CARD_WIDTH - 40, center_y - CARD_HEIGHT // 2, True, True)
        
        # Direction indicator
        direction_text = ">> Clockwise >>" if self.direction == 1 else "<< Counter <<"
        direction_surf = FONT_SMALL.render(direction_text, True, COLORS['ACCENT_CYAN'])
        self.screen.blit(direction_surf, (center_x - direction_surf.get_width() // 2, center_y - CARD_HEIGHT // 2 - 40))
    
    def render_color_selector(self):
        """Render color selection UI for wild cards."""
        # Darken background
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = FONT_LARGE.render("Choose a Color", True, COLORS['WHITE'])
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, WINDOW_HEIGHT // 2 - 80))
        
        # Color buttons
        colors = [("RED", COLORS['RED']), ("GRE", COLORS['GRE']), 
                  ("BLU", COLORS['BLU']), ("YEL", COLORS['YEL'])]
        
        for i, (name, color) in enumerate(colors):
            x = WINDOW_WIDTH // 2 - 150 + i * 80
            y = WINDOW_HEIGHT // 2
            rect = pygame.Rect(x, y, 60, 60)
            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            pygame.draw.rect(self.screen, COLORS['WHITE'], rect, 3, border_radius=10)
    
    def render_game_over(self):
        """Render game over screen."""
        # Darken background
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Winner text
        if self.winner == 0:
            text = "YOU WIN!"
            color = COLORS['GOLD']
        else:
            text = f"{PLAYER_POSITIONS[self.winner]['name']} Wins!"
            color = PLAYER_POSITIONS[self.winner]['color']
        
        title = FONT_LARGE.render(text, True, color)
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, WINDOW_HEIGHT // 2 - 80))
        
        # New game button
        button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 20, 200, 50)
        pygame.draw.rect(self.screen, COLORS['ACCENT_CYAN'], button_rect, border_radius=10)
        
        button_text = FONT_MEDIUM.render("New Game", True, COLORS['WHITE'])
        self.screen.blit(button_text, (button_rect.centerx - button_text.get_width() // 2,
                                       button_rect.centery - button_text.get_height() // 2))
    
    def render_message(self):
        """Render temporary message."""
        if self.message_timer > 0:
            # Message box
            msg_surf = FONT_MEDIUM.render(self.message, True, COLORS['WHITE'])
            padding = 20
            box_rect = pygame.Rect(
                WINDOW_WIDTH // 2 - msg_surf.get_width() // 2 - padding,
                150 - padding,
                msg_surf.get_width() + padding * 2,
                msg_surf.get_height() + padding * 2
            )
            
            # Background
            pygame.draw.rect(self.screen, COLORS['PANEL_BG'], box_rect, border_radius=10)
            pygame.draw.rect(self.screen, COLORS['GOLD'], box_rect, 2, border_radius=10)
            
            self.screen.blit(msg_surf, (box_rect.x + padding, box_rect.y + padding))
    
    def render(self):
        """Render the entire game."""
        self.render_background()
        
        # Render close button (top-right corner)
        close_rect = pygame.Rect(WINDOW_WIDTH - 45, 10, 35, 35)
        pygame.draw.rect(self.screen, (239, 68, 68), close_rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 100, 100), close_rect, 2, border_radius=8)
        x_text = FONT_MEDIUM.render("X", True, COLORS['WHITE'])
        self.screen.blit(x_text, (close_rect.centerx - x_text.get_width() // 2,
                                   close_rect.centery - x_text.get_height() // 2 - 2))
        
        # Render all player hands
        for i in range(self.num_players):
            self.render_player_hand(i)
        
        # Render center
        self.render_center()
        
        # Render message
        self.render_message()
        
        # Render color selector if active
        if self.selecting_color:
            self.render_color_selector()
        
        # Render game over if finished
        if self.game_over:
            self.render_game_over()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_n:
                        self.new_game()
            
            self.update()
            self.render()
            self.clock.tick(FPS)
        
        pygame.quit()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multiplayer UNO")
    parser.add_argument("--players", type=int, default=4, choices=[3, 4],
                       help="Number of players (3 or 4)")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║            MULTIPLAYER UNO - {args.players} Players                   ║
╠══════════════════════════════════════════════════════════╣
║  Controls:                                               ║
║    - Click card to play                                  ║
║    - Click deck to draw (when no cards playable)         ║
║    - Press N for new game                                ║
║    - Press ESC to quit                                   ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    game = MultiplayerUnoGame(num_players=args.players)
    game.run()


if __name__ == "__main__":
    main()
