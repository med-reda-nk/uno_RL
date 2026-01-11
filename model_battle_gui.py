"""
Model vs Model Battle GUI for UNO
Watch trained AI models compete against each other and save results.
"""

import pygame
import sys
import random
import os
import csv
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Tuple
import numpy as np

# Try to import ML libraries
try:
    from stable_baselines3 import PPO, DQN, A2C
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_AVAILABLE = True
except ImportError:
    RECURRENT_AVAILABLE = False

# Initialize Pygame
pygame.init()
pygame.font.init()

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

WINDOW_WIDTH = 1050
WINDOW_HEIGHT = 700
FPS = 60

# Ultra-modern color palette - Premium Design
COLORS = {
    # Card colors - vibrant
    'RED': (239, 68, 68),
    'GRE': (16, 185, 129),
    'BLU': (59, 130, 246),
    'YEL': (250, 204, 21),
    'WILD': (24, 24, 36),
    
    # UI colors
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (100, 100, 100),
    'LIGHT_GRAY': (226, 232, 240),
    'DARK_GRAY': (30, 41, 59),
    'MUTED': (148, 163, 184),
    
    # Background - deep space theme
    'BG_TOP': (8, 12, 28),
    'BG_BOTTOM': (18, 26, 48),
    'PANEL': (18, 28, 50),
    'PANEL_BORDER': (60, 75, 100),
    
    # Accent colors - neon palette
    'ACCENT': (99, 102, 241),
    'ACCENT_HOVER': (129, 140, 248),
    'ACCENT_CYAN': (34, 211, 238),
    'ACCENT_PURPLE': (192, 132, 252),
    'ACCENT_PINK': (244, 114, 182),
    'ACCENT_BLUE': (56, 189, 248),
    'ACCENT_GREEN': (74, 222, 128),
    'ACCENT_ORANGE': (251, 146, 60),
    'SUCCESS': (34, 197, 94),
    'SUCCESS_HOVER': (74, 222, 128),
    'DANGER': (239, 68, 68),
    'DANGER_HOVER': (248, 113, 113),
    'WARNING': (250, 204, 21),
    'GOLD': (251, 191, 36),
    'SILVER': (203, 213, 225),
    'BRONZE': (217, 119, 6),
}

CARD_WIDTH = 75
CARD_HEIGHT = 105
CARD_RADIUS = 14

# Modern Typography - Elegant sizing
try:
    FONT_TITLE = pygame.font.SysFont('segoeui', 42, bold=True)
    FONT_LARGE = pygame.font.SysFont('segoeui', 28, bold=True)
    FONT_MEDIUM = pygame.font.SysFont('segoeui', 20)
    FONT_SMALL = pygame.font.SysFont('segoeui', 16)
    FONT_TINY = pygame.font.SysFont('segoeui', 13)
    FONT_MICRO = pygame.font.SysFont('segoeui', 11)
    FONT_CARD = pygame.font.SysFont('segoeui', 22, bold=True)
    FONT_CARD_LARGE = pygame.font.SysFont('segoeui', 30, bold=True)
except:
    FONT_TITLE = pygame.font.Font(None, 46)
    FONT_LARGE = pygame.font.Font(None, 32)
    FONT_MEDIUM = pygame.font.Font(None, 24)
    FONT_SMALL = pygame.font.Font(None, 18)
    FONT_TINY = pygame.font.Font(None, 15)
    FONT_MICRO = pygame.font.Font(None, 13)
    FONT_CARD = pygame.font.Font(None, 26)
    FONT_CARD_LARGE = pygame.font.Font(None, 34)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def draw_gradient_bg(surface, color1, color2):
    """Draw modern vertical gradient with smooth easing."""
    height = surface.get_height()
    width = surface.get_width()
    for y in range(height):
        progress = y / height
        # Smooth cubic easing
        eased = progress * progress * (3 - 2 * progress)
        r = int(color1[0] + eased * (color2[0] - color1[0]))
        g = int(color1[1] + eased * (color2[1] - color1[1]))
        b = int(color1[2] + eased * (color2[2] - color1[2]))
        pygame.draw.line(surface, (r, g, b), (0, y), (width, y))


def draw_ambient_orbs(surface, animation_offset):
    """Draw animated ambient glow orbs."""
    import math
    orb_configs = [
        (COLORS['ACCENT_PURPLE'], 0.12, 180, 0.4),
        (COLORS['ACCENT_CYAN'], 0.10, 160, 0.5),
        (COLORS['ACCENT_BLUE'], 0.15, 140, 0.3),
        (COLORS['ACCENT_PINK'], 0.08, 120, 0.6),
    ]
    
    for i, (color, speed, radius, vert_scale) in enumerate(orb_configs):
        angle = math.radians(animation_offset * speed + i * 90)
        cx = surface.get_width() / 2 + math.cos(angle) * (200 + i * 50)
        cy = surface.get_height() / 2 + math.sin(angle) * (100 + i * 30) * vert_scale
        
        glow_surface = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
        for rad in range(radius, 10, -5):
            alpha = int(15 * (radius - rad) / radius)
            pygame.draw.circle(glow_surface, (*color[:3], alpha), 
                             (radius * 1.5, radius * 1.5), rad)
        
        surface.blit(glow_surface, (int(cx - radius * 1.5), int(cy - radius * 1.5)))


def draw_grid_overlay(surface):
    """Draw subtle modern grid pattern."""
    grid_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
    spacing = 60
    alpha = 6
    for gx in range(0, surface.get_width(), spacing):
        pygame.draw.line(grid_surface, (255, 255, 255, alpha), (gx, 0), (gx, surface.get_height()))
    for gy in range(0, surface.get_height(), spacing):
        pygame.draw.line(grid_surface, (255, 255, 255, alpha), (0, gy), (surface.get_width(), gy))
    surface.blit(grid_surface, (0, 0))


# =============================================================================
# CARD & DECK CLASSES
# =============================================================================

class Card:
    def __init__(self, color: str, value):
        self.color = color
        self.value = value
        
    def can_play_on(self, other: 'Card') -> bool:
        if self.value in ["COL", "PL4"]:
            return True
        return self.color == other.color or self.value == other.value
    
    def get_display_value(self) -> str:
        if self.value == "SKI":
            return "⊘"
        elif self.value == "REV":
            return "⟲"
        elif self.value == "PL2":
            return "+2"
        elif self.value == "PL4":
            return "+4"
        elif self.value == "COL":
            return "✦"
        else:
            return str(self.value)
    
    def get_color_rgb(self):
        if self.color in COLORS:
            return COLORS[self.color]
        return COLORS['WILD']
    
    def render(self, surface, x: int, y: int, face_up: bool = True, scale: float = 1.0):
        """Render card with premium styling (shadows, gradients, oval)."""
        w = int(CARD_WIDTH * scale)
        h = int(CARD_HEIGHT * scale)
        margin = 6
        
        card_surface = pygame.Surface((w + margin * 2, h + margin * 2), pygame.SRCALPHA)
        
        # Soft shadow layers
        for i in range(3):
            shadow_alpha = 25 - i * 7
            shadow_rect = pygame.Rect(margin + i, margin + 2 + i, w, h)
            pygame.draw.rect(card_surface, (0, 0, 0, shadow_alpha), shadow_rect, 
                           border_radius=int(CARD_RADIUS * scale))
        
        card_rect = pygame.Rect(margin, margin, w, h)
        
        if face_up:
            base_color = self.get_color_rgb()
            pygame.draw.rect(card_surface, base_color, card_rect, 
                           border_radius=int(CARD_RADIUS * scale))
            
            # Top gradient highlight
            highlight_h = int(h * 0.28)
            highlight_surf = pygame.Surface((w - 4, highlight_h), pygame.SRCALPHA)
            for hy in range(highlight_h):
                alpha = int(35 * (1 - hy / highlight_h))
                pygame.draw.line(highlight_surf, (255, 255, 255, alpha), (0, hy), (w - 4, hy))
            card_surface.blit(highlight_surf, (margin + 2, margin + 2))
            
            # Inner white oval (classic UNO style)
            inner_margin = int(5 * scale)
            oval_surface = pygame.Surface((w, h), pygame.SRCALPHA)
            oval_rect = pygame.Rect(inner_margin, int(14 * scale), 
                                   w - 2 * inner_margin, h - int(28 * scale))
            pygame.draw.ellipse(oval_surface, (255, 255, 255, 230), oval_rect)
            
            rotated_oval = pygame.transform.rotate(oval_surface, 25)
            oval_pos = (margin + w // 2 - rotated_oval.get_width() // 2,
                       margin + h // 2 - rotated_oval.get_height() // 2)
            card_surface.blit(rotated_oval, oval_pos)
            
            # Value text with shadow
            display_val = self.get_display_value()
            font = FONT_CARD_LARGE if scale >= 0.8 else FONT_CARD
            
            text_shadow = font.render(display_val, True, (0, 0, 0, 60))
            shadow_rect = text_shadow.get_rect(center=(margin + w // 2 + 1, margin + h // 2 + 1))
            card_surface.blit(text_shadow, shadow_rect)
            
            text_main = font.render(display_val, True, base_color)
            text_rect = text_main.get_rect(center=(margin + w // 2, margin + h // 2))
            card_surface.blit(text_main, text_rect)
            
            # Corner values
            corner_text = FONT_TINY.render(display_val, True, COLORS['WHITE'])
            card_surface.blit(corner_text, (margin + int(5 * scale), margin + int(5 * scale)))
            
            rotated_corner = pygame.transform.rotate(corner_text, 180)
            card_surface.blit(rotated_corner, (margin + w - int(14 * scale), margin + h - int(16 * scale)))
        else:
            # Card back design
            pygame.draw.rect(card_surface, COLORS['WILD'], card_rect, 
                           border_radius=int(CARD_RADIUS * scale))
            
            inner_border = pygame.Rect(margin + 3, margin + 3, w - 6, h - 6)
            pygame.draw.rect(card_surface, (50, 50, 70), inner_border, 
                           width=2, border_radius=int(CARD_RADIUS * scale) - 2)
            
            oval_rect = pygame.Rect(margin + int(7 * scale), margin + int(12 * scale), 
                                   w - int(14 * scale), h - int(24 * scale))
            pygame.draw.ellipse(card_surface, COLORS['RED'], oval_rect)
            
            inner_oval = pygame.Rect(margin + int(10 * scale), margin + int(18 * scale), 
                                    w - int(20 * scale), h - int(36 * scale))
            pygame.draw.ellipse(card_surface, COLORS['YEL'], inner_oval)
            
            uno_text = FONT_TINY.render("UNO", True, COLORS['RED'])
            uno_rotated = pygame.transform.rotate(uno_text, -25)
            text_rect = uno_rotated.get_rect(center=(margin + w // 2, margin + h // 2))
            card_surface.blit(uno_rotated, text_rect)
        
        surface.blit(card_surface, (x - margin, y - margin))
    
    def __repr__(self):
        return f"{self.color}:{self.value}"


class Deck:
    def __init__(self):
        self.cards = []
        self.discard_pile = []
        self._create_deck()
        self.shuffle()
    
    def _create_deck(self):
        colors = ["RED", "GRE", "BLU", "YEL"]
        for color in colors:
            self.cards.append(Card(color, 0))
            for value in range(1, 10):
                self.cards.append(Card(color, value))
                self.cards.append(Card(color, value))
            for special in ["SKI", "REV", "PL2"]:
                self.cards.append(Card(color, special))
                self.cards.append(Card(color, special))
        for _ in range(4):
            self.cards.append(Card("WILD", "COL"))
            self.cards.append(Card("WILD", "PL4"))
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def draw(self) -> Optional[Card]:
        if not self.cards:
            if self.discard_pile:
                self.cards = self.discard_pile[:-1]
                self.discard_pile = [self.discard_pile[-1]]
                self.shuffle()
            else:
                return None
        return self.cards.pop() if self.cards else None
    
    def discard(self, card: Card):
        self.discard_pile.append(card)


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class ModelWrapper:
    """Wrapper for different model types with unified interface."""
    
    def __init__(self, path: str, model_type: str, name: str):
        self.path = path
        self.model_type = model_type
        self.name = name
        self.model = None
        self.is_recurrent = model_type == "recurrentppo"
        self.lstm_states = None
        self.loaded = False
        
    def load(self) -> bool:
        try:
            if self.model_type == "recurrentppo" and RECURRENT_AVAILABLE:
                self.model = RecurrentPPO.load(self.path)
            elif self.model_type == "ppo" and SB3_AVAILABLE:
                self.model = PPO.load(self.path)
            elif self.model_type == "dqn" and SB3_AVAILABLE:
                self.model = DQN.load(self.path)
            elif self.model_type == "a2c" and SB3_AVAILABLE:
                self.model = A2C.load(self.path)
            elif self.model_type == "random":
                self.model = "random"
            else:
                return False
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error loading {self.name}: {e}")
            return False
    
    def reset(self):
        self.lstm_states = None
    
    def predict(self, obs: np.ndarray) -> int:
        if self.model == "random":
            return random.randint(0, 8)
        
        if self.is_recurrent:
            episode_start = np.array([self.lstm_states is None])
            action, self.lstm_states = self.model.predict(
                obs, state=self.lstm_states, episode_start=episode_start, deterministic=True
            )
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


# =============================================================================
# BATTLE ENVIRONMENT
# =============================================================================

class BattleEnv:
    """Environment for model vs model battles - supports 2-4 players."""
    
    ACTION_NAMES = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.deck = None
        self.hands = [[] for _ in range(num_players)]
        self.open_card = None
        self.current_player = 0
        self.winner = None
        self.turn_count = 0
        self.max_turns = 500
        self.game_log = []
        self.direction = 1  # 1 for clockwise, -1 for counter-clockwise
    
    def reset(self):
        self.deck = Deck()
        self.hands = [[] for _ in range(self.num_players)]
        self.winner = None
        self.turn_count = 0
        self.game_log = []
        self.direction = 1
        
        # Deal 7 cards each
        for _ in range(7):
            for i in range(self.num_players):
                card = self.deck.draw()
                if card:
                    self.hands[i].append(card)
        
        # Draw open card (must be number)
        self.open_card = self.deck.draw()
        while self.open_card and self.open_card.value not in range(0, 10):
            self.deck.cards.insert(0, self.open_card)
            self.deck.shuffle()
            self.open_card = self.deck.draw()
        
        self.current_player = 0
        return self.get_observation(0)
    
    def get_observation(self, player: int) -> np.ndarray:
        hand = self.hands[player]
        
        # One-hot open card color
        color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        color_vec = [0, 0, 0, 0]
        if self.open_card and self.open_card.color in color_map:
            color_vec[color_map[self.open_card.color]] = 1
        
        # Card counts
        norm_cards = {"RED": 0, "GRE": 0, "BLU": 0, "YEL": 0}
        spec_cards = {"SKI": 0, "REV": 0, "PL2": 0}
        wild_cards = {"PL4": 0, "COL": 0}
        play_norm = {"RED#": 0, "GRE#": 0, "BLU#": 0, "YEL#": 0}
        
        for card in hand:
            if card.color in norm_cards and card.value in range(0, 10):
                norm_cards[card.color] = min(norm_cards[card.color] + 1, 2)
            if card.value in spec_cards:
                spec_cards[card.value] = min(spec_cards[card.value] + 1, 1)
            if card.value in wild_cards:
                wild_cards[card.value] = min(wild_cards[card.value] + 1, 1)
            
            # Check if playable
            if card.can_play_on(self.open_card):
                if card.color in norm_cards and card.value in range(0, 10):
                    play_norm[card.color + "#"] = 1
        
        card_values = []
        for c in ["RED", "GRE", "BLU", "YEL"]:
            card_values.append(norm_cards[c] / 2.0)
        for s in ["SKI", "REV", "PL2"]:
            card_values.append(spec_cards[s])
        for w in ["PL4", "COL"]:
            card_values.append(wild_cards[w])
        for c in ["RED", "GRE", "BLU", "YEL"]:
            card_values.append(play_norm[c + "#"])
        
        obs = np.array(color_vec + card_values, dtype=np.float32)
        if len(obs) < 17:
            obs = np.pad(obs, (0, 17 - len(obs)))
        return obs[:17]
    
    def get_playable(self, player: int) -> List[Card]:
        return [c for c in self.hands[player] if c.can_play_on(self.open_card)]
    
    def find_card_for_action(self, action: int, player: int) -> Optional[Card]:
        action_name = self.ACTION_NAMES[action]
        hand = self.hands[player]
        playable = self.get_playable(player)
        
        if action_name in ["COL", "PL4"]:
            for card in hand:
                if card.value == action_name:
                    return card
        elif action_name in ["RED", "GRE", "BLU", "YEL"]:
            for card in playable:
                if card.color == action_name and card.value in range(0, 10):
                    return card
        elif action_name in ["SKI", "REV", "PL2"]:
            for card in playable:
                if card.value == action_name:
                    return card
        
        # Fallback to any playable
        return playable[0] if playable else None
    
    def step(self, action: int) -> Tuple[bool, str]:
        """Execute one turn. Returns (game_over, event_description)."""
        player = self.current_player
        self.turn_count += 1
        
        # Calculate next player based on direction
        def next_player(p, skip=1):
            return (p + self.direction * skip) % self.num_players
        
        playable = self.get_playable(player)
        
        if not playable:
            # Draw card
            drawn = self.deck.draw()
            if drawn:
                self.hands[player].append(drawn)
                event = f"Player {player+1} draws"
            else:
                event = f"Player {player+1} passes (no cards)"
            self.current_player = next_player(player)
            self.game_log.append(event)
            return False, event
        
        card = self.find_card_for_action(action, player)
        if not card:
            card = playable[0]
        
        self.hands[player].remove(card)
        
        # Handle wild cards
        if card.value in ["COL", "PL4"]:
            colors = [c.color for c in self.hands[player] if c.color in ["RED", "GRE", "BLU", "YEL"]]
            card.color = max(set(colors), key=colors.count) if colors else random.choice(["RED", "GRE", "BLU", "YEL"])
        
        event = f"Player {player+1} plays {card.color} {card.value}"
        
        # Handle special effects
        target = next_player(player)
        
        if card.value == "PL4":
            for _ in range(4):
                drawn = self.deck.draw()
                if drawn:
                    self.hands[target].append(drawn)
            event += f" (+4 to P{target+1})"
            # Skip the player who got +4
            self.current_player = next_player(player, skip=2) if self.num_players > 2 else player
        elif card.value == "PL2":
            for _ in range(2):
                drawn = self.deck.draw()
                if drawn:
                    self.hands[target].append(drawn)
            event += f" (+2 to P{target+1})"
            # Skip the player who got +2
            self.current_player = next_player(player, skip=2) if self.num_players > 2 else player
        elif card.value == "SKI":
            event += f" (skip P{target+1})"
            self.current_player = next_player(player, skip=2) if self.num_players > 2 else player
        elif card.value == "REV":
            if self.num_players == 2:
                # In 2-player, reverse acts as skip
                event += " (skip)"
                self.current_player = player
            else:
                # In 3-4 player, reverse changes direction
                self.direction *= -1
                event += " (reverse)"
                self.current_player = next_player(player)
        else:
            self.current_player = next_player(player)
        
        self.open_card = card
        self.deck.discard(card)
        self.game_log.append(event)
        
        # Check win
        if len(self.hands[player]) == 0:
            self.winner = player
            return True, event
        
        # Check max turns
        if self.turn_count >= self.max_turns:
            # Winner is player with fewest cards
            min_cards = min(len(h) for h in self.hands)
            candidates = [i for i, h in enumerate(self.hands) if len(h) == min_cards]
            self.winner = random.choice(candidates)
            return True, "Max turns reached"
        
        return False, event


# =============================================================================
# GUI COMPONENTS
# =============================================================================

class Button:
    def __init__(self, x, y, w, h, text, color=COLORS['ACCENT'], 
                 hover_color=None, text_color=COLORS['WHITE'], icon=None, gradient=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color or tuple(min(c + 30, 255) for c in color)
        self.text_color = text_color
        self.icon = icon
        self.hover = False
        self.enabled = True
        self.gradient = gradient
        self.glow_alpha = 0
        self.ripple_alpha = 0
        
    def draw(self, screen):
        color = self.hover_color if self.hover and self.enabled else self.color
        if not self.enabled:
            color = COLORS['GRAY']
        
        # Animate glow
        target_glow = 180 if self.hover and self.enabled else 0
        self.glow_alpha += (target_glow - self.glow_alpha) * 0.15
        
        # 5-layer glow effect on hover
        if self.glow_alpha > 5:
            for i in range(5, 0, -1):
                glow_rect = self.rect.inflate(i * 4, i * 4)
                glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                alpha = int(self.glow_alpha * (6 - i) / 20)
                pygame.draw.rect(glow_surf, (*color[:3], alpha), 
                               glow_surf.get_rect(), border_radius=12 + i * 2)
                screen.blit(glow_surf, glow_rect.topleft)
        
        # Multi-layer shadow
        for offset, alpha in [(4, 20), (2, 35)]:
            shadow_rect = self.rect.move(0, offset)
            shadow_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surf, (0, 0, 0, alpha), shadow_surf.get_rect(), border_radius=10)
            screen.blit(shadow_surf, shadow_rect.topleft)
        
        # Button body with gradient
        btn_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        if self.gradient and self.enabled:
            for y in range(self.rect.height):
                ratio = y / self.rect.height
                r = int(color[0] * (1 - ratio * 0.3))
                g = int(color[1] * (1 - ratio * 0.3))
                b = int(color[2] * (1 - ratio * 0.3))
                pygame.draw.line(btn_surf, (r, g, b), (0, y), (self.rect.width, y))
            pygame.draw.rect(btn_surf, (0, 0, 0, 0), btn_surf.get_rect(), border_radius=10)
        else:
            pygame.draw.rect(btn_surf, color, btn_surf.get_rect(), border_radius=10)
        screen.blit(btn_surf, self.rect.topleft)
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        
        # Top highlight gradient
        if self.enabled:
            highlight = pygame.Rect(0, 0, self.rect.width - 4, self.rect.height // 2 - 2)
            hs = pygame.Surface((highlight.width, highlight.height), pygame.SRCALPHA)
            for y in range(highlight.height):
                alpha = int(40 * (1 - y / highlight.height))
                pygame.draw.line(hs, (255, 255, 255, alpha), (2, y), (highlight.width - 2, y))
            screen.blit(hs, (self.rect.x + 2, self.rect.y + 2))
        
        # Border glow
        border_color = (*color[:3], 60) if self.enabled else (100, 100, 100, 40)
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=10)
        
        # Inner highlight line
        if self.enabled and self.hover:
            line_rect = pygame.Rect(self.rect.x + 8, self.rect.y + 3, self.rect.width - 16, 1)
            pygame.draw.rect(screen, (*COLORS['WHITE'][:3], 80), line_rect)
        
        # Text with icon
        display_text = f"{self.icon} {self.text}" if self.icon else self.text
        text_surf = FONT_SMALL.render(display_text, True, 
                                     self.text_color if self.enabled else COLORS['MUTED'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        
        # Text shadow
        if self.enabled:
            shadow_text = FONT_SMALL.render(display_text, True, (0, 0, 0))
            screen.blit(shadow_text, (text_rect.x + 1, text_rect.y + 1))
        
        screen.blit(text_surf, text_rect)
    
    def update(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)
    
    def clicked(self, mouse_pos) -> bool:
        return self.enabled and self.rect.collidepoint(mouse_pos)


class ModelSelector:
    def __init__(self, x, y, w, h, title, models, accent_color=COLORS['ACCENT']):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.models = models
        self.selected_idx = 0
        self.scroll_offset = 0
        self.item_height = 30
        self.accent_color = accent_color
        self.hover_idx = -1
        
    def draw(self, screen):
        # Glassmorphism panel
        panel_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        
        # Multi-layer glass effect
        pygame.draw.rect(panel_surf, (15, 23, 42, 200), panel_surf.get_rect(), border_radius=14)
        pygame.draw.rect(panel_surf, (30, 41, 59, 80), panel_surf.get_rect(), border_radius=14)
        
        # Top shine
        shine_rect = pygame.Rect(4, 4, self.rect.width - 8, 40)
        shine_surf = pygame.Surface((shine_rect.width, shine_rect.height), pygame.SRCALPHA)
        for y in range(shine_rect.height):
            alpha = int(20 * (1 - y / shine_rect.height))
            pygame.draw.line(shine_surf, (255, 255, 255, alpha), (0, y), (shine_rect.width, y))
        panel_surf.blit(shine_surf, shine_rect.topleft)
        
        screen.blit(panel_surf, self.rect.topleft)
        
        # Glowing border with accent
        for i in range(3, 0, -1):
            border_rect = self.rect.inflate(i * 2, i * 2)
            alpha = int(30 / i)
            pygame.draw.rect(screen, (*self.accent_color[:3], alpha), border_rect, 2, border_radius=14 + i)
        pygame.draw.rect(screen, self.accent_color, self.rect, 2, border_radius=14)
        
        # Title with glow
        title_surf = FONT_MEDIUM.render(self.title, True, self.accent_color)
        glow_surf = FONT_MEDIUM.render(self.title, True, (*self.accent_color[:3], 60))
        screen.blit(glow_surf, (self.rect.x + 13, self.rect.y + 11))
        screen.blit(title_surf, (self.rect.x + 12, self.rect.y + 10))
        
        # Model list
        list_y = self.rect.y + 45
        visible_items = (self.rect.height - 55) // self.item_height
        
        for i in range(min(visible_items, len(self.models))):
            idx = i + self.scroll_offset
            if idx >= len(self.models):
                break
            
            model = self.models[idx]
            item_rect = pygame.Rect(self.rect.x + 8, list_y + i * self.item_height, 
                                   self.rect.width - 16, self.item_height - 4)
            
            item_surf = pygame.Surface((item_rect.width, item_rect.height), pygame.SRCALPHA)
            
            if idx == self.selected_idx:
                # Selected - accent gradient
                for y in range(item_rect.height):
                    ratio = y / item_rect.height
                    r = int(self.accent_color[0] * (1 - ratio * 0.3))
                    g = int(self.accent_color[1] * (1 - ratio * 0.3))
                    b = int(self.accent_color[2] * (1 - ratio * 0.3))
                    pygame.draw.line(item_surf, (r, g, b), (0, y), (item_rect.width, y))
                pygame.draw.rect(item_surf, self.accent_color, item_surf.get_rect(), 1, border_radius=8)
                text_color = COLORS['WHITE']
            elif idx == self.hover_idx:
                # Hovered
                pygame.draw.rect(item_surf, (51, 65, 85, 200), item_surf.get_rect(), border_radius=8)
                text_color = COLORS['WHITE']
            else:
                # Normal
                pygame.draw.rect(item_surf, (30, 41, 59, 150), item_surf.get_rect(), border_radius=8)
                text_color = COLORS['LIGHT_GRAY']
            
            screen.blit(item_surf, item_rect.topleft)
            
            # Model name
            name = model.name[:22] + "..." if len(model.name) > 25 else model.name
            text = FONT_TINY.render(name, True, text_color)
            screen.blit(text, (item_rect.x + 10, item_rect.y + 7))
    
    def handle_click(self, mouse_pos):
        if not self.rect.collidepoint(mouse_pos):
            return
        
        list_y = self.rect.y + 45
        rel_y = mouse_pos[1] - list_y
        if rel_y < 0:
            return
        
        clicked_idx = rel_y // self.item_height + self.scroll_offset
        if 0 <= clicked_idx < len(self.models):
            self.selected_idx = clicked_idx
    
    def handle_hover(self, mouse_pos):
        if not self.rect.collidepoint(mouse_pos):
            self.hover_idx = -1
            return
        
        list_y = self.rect.y + 45
        rel_y = mouse_pos[1] - list_y
        if rel_y < 0:
            self.hover_idx = -1
            return
        
        hover_idx = rel_y // self.item_height + self.scroll_offset
        self.hover_idx = hover_idx if 0 <= hover_idx < len(self.models) else -1
    
    def handle_scroll(self, direction):
        max_scroll = max(0, len(self.models) - (self.rect.height - 55) // self.item_height)
        self.scroll_offset = max(0, min(max_scroll, self.scroll_offset + direction))
    
    def get_selected(self) -> Optional[ModelWrapper]:
        if 0 <= self.selected_idx < len(self.models):
            return self.models[self.selected_idx]
        return None


# =============================================================================
# MAIN GUI
# =============================================================================

class ModelBattleGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("UNO Model Battle Arena")
        self.clock = pygame.time.Clock()
        
        # Pre-render gradient background
        self.bg_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        draw_gradient_bg(self.bg_surface, COLORS['BG_TOP'], COLORS['BG_BOTTOM'])
        
        # Animation state
        self.animation_offset = 0
        
        self.state = "SELECT"
        self.models = self.discover_models()
        self.num_players = 2  # Default to 2 players
        
        # Model selectors with accent colors - support up to 4 players
        selector_colors = [COLORS['ACCENT'], COLORS['DANGER'], 
                          COLORS['SUCCESS'], COLORS['WARNING']]
        selector_width = 185
        
        self.selectors = []
        for i in range(4):
            x = 25 + i * (selector_width + 12)
            selector = ModelSelector(x, 110, selector_width, 280, 
                                    f"[{i+1}] Player {i+1}", 
                                    self.models, selector_colors[i])
            if i < len(self.models):
                selector.selected_idx = min(i, len(self.models) - 1)
            self.selectors.append(selector)
        
        # Legacy references for compatibility
        self.selector1 = self.selectors[0]
        self.selector2 = self.selectors[1]
        
        # Player count buttons
        self.player_2_btn = Button(25, 65, 80, 32, "2P", COLORS['ACCENT'])
        self.player_3_btn = Button(115, 65, 80, 32, "3P", COLORS['GRAY'])
        self.player_4_btn = Button(205, 65, 80, 32, "4P", COLORS['GRAY'])
        
        # Buttons with icons - positioned below selectors
        btn_y = 405
        self.start_button = Button(25, btn_y, 130, 38, "Battle", 
                                   COLORS['SUCCESS'], COLORS['SUCCESS_HOVER'], icon=">>")
        self.batch_button = Button(170, btn_y, 130, 38, "Batch 100", 
                                   COLORS['ACCENT'], COLORS['ACCENT_HOVER'], icon="#")
        self.back_button = Button(15, 12, 90, 34, "Back", COLORS['GRAY'], icon="<")
        self.save_button = Button(WINDOW_WIDTH - 190, 12, 120, 34, "Save CSV", 
                                  COLORS['SUCCESS'], icon="*")
        self.close_button = Button(WINDOW_WIDTH - 55, 12, 42, 34, "X", 
                                   COLORS['DANGER'], COLORS['DANGER_HOVER'])
        
        # Battle state
        self.env = BattleEnv(num_players=2)
        self.battle_models = []  # List of models for battle
        self.model1 = None
        self.model2 = None
        self.battle_speed = 30  # frames per turn
        self.frame_count = 0
        self.game_over = False
        self.last_event = ""
        self.auto_play = True
        
        # Results
        self.results = []
        self.batch_mode = False
        self.batch_games = 0
        self.batch_target = 100
        self.wins = [0, 0, 0, 0]  # Support 4 players
        
    def discover_models(self) -> List[ModelWrapper]:
        """Find all available models."""
        models = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        # Add random baseline
        models.append(ModelWrapper("", "random", "Random Agent"))
        
        model_files = [
            ("selfplay_champion.zip", "recurrentppo", "[NEW] Self-Play Champion"),
            ("best_recurrent_ppo_uno.zip", "recurrentppo", "[1st] Best Recurrent PPO"),
            ("optimal_recurrent_ppo.zip", "recurrentppo", "[2nd] Optimal Recurrent PPO"),
            ("sb3_recurrentppo_uno.zip", "recurrentppo", "[3rd] SB3 Recurrent PPO"),
            ("recurrent_ppo_uno.zip", "recurrentppo", "Recurrent PPO"),
            ("enhanced_rppo.zip", "recurrentppo", "Enhanced RPPO"),
            ("best_enhanced_rppo/best_model.zip", "recurrentppo", "Best Enhanced RPPO"),
            ("best_optimal_rppo/best_model.zip", "recurrentppo", "Best Optimal RPPO"),
            ("sb3_ppo_uno.zip", "ppo", "PPO"),
            ("best_ppo_uno.zip", "ppo", "Best PPO"),
            ("sb3_dqn_uno.zip", "dqn", "DQN"),
            ("sb3_a2c_uno.zip", "a2c", "A2C"),
            ("best_model.zip", "ppo", "Best Model"),
        ]
        
        for filename, model_type, name in model_files:
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                models.append(ModelWrapper(path, model_type, name))
        
        return models
    
    def set_num_players(self, num: int):
        """Set the number of players (2-4)."""
        self.num_players = max(2, min(4, num))
        self.env = BattleEnv(num_players=self.num_players)
        self.wins = [0] * self.num_players
        
        # Update button colors
        self.player_2_btn.color = COLORS['ACCENT'] if num == 2 else COLORS['GRAY']
        self.player_3_btn.color = COLORS['ACCENT'] if num == 3 else COLORS['GRAY']
        self.player_4_btn.color = COLORS['ACCENT'] if num == 4 else COLORS['GRAY']
    
    def start_battle(self, batch=False):
        """Initialize a battle between selected models."""
        # Get models for all active players
        self.battle_models = []
        for i in range(self.num_players):
            model = self.selectors[i].get_selected()
            if not model:
                return
            self.battle_models.append(model)
        
        # Legacy compatibility
        self.model1 = self.battle_models[0]
        self.model2 = self.battle_models[1] if len(self.battle_models) > 1 else None
        
        # Load all models
        for model in self.battle_models:
            if not model.loaded:
                if not model.load():
                    print(f"Failed to load {model.name}")
                    return
            model.reset()
        
        self.env = BattleEnv(num_players=self.num_players)
        self.env.reset()
        self.game_over = False
        self.last_event = "Battle started!"
        self.frame_count = 0
        
        if batch:
            self.batch_mode = True
            self.batch_games = 0
            self.wins = [0] * self.num_players
            self.results = []
        else:
            self.batch_mode = False
        
        self.state = "BATTLE"
    
    def run_turn(self):
        """Execute one turn in the battle."""
        if self.game_over:
            return
        
        current = self.env.current_player
        model = self.battle_models[current]
        
        obs = self.env.get_observation(current)
        action = model.predict(obs)
        
        self.game_over, self.last_event = self.env.step(action)
        
        if self.game_over:
            winner_idx = self.env.winner
            self.wins[winner_idx] += 1
            
            result = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_players": self.num_players,
                "winner": self.battle_models[winner_idx].name,
                "winner_idx": winner_idx + 1,
                "turns": self.env.turn_count,
            }
            # Add model names and final card counts
            for i, model in enumerate(self.battle_models):
                result[f"model{i+1}"] = model.name
                result[f"model{i+1}_cards"] = len(self.env.hands[i])
            
            self.results.append(result)
            
            if self.batch_mode:
                self.batch_games += 1
                if self.batch_games < self.batch_target:
                    # Start next game - reset all models
                    for model in self.battle_models:
                        model.reset()
                    self.env.reset()
                    self.game_over = False
                else:
                    self.state = "RESULTS"
    
    def save_results(self):
        """Save battle results to CSV."""
        if not self.results:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results/battle_{timestamp}.csv"
        
        os.makedirs("comparison_results", exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        return filename
    
    def draw_select_screen(self):
        """Draw model selection screen with modern background."""
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Animated effects
        draw_ambient_orbs(self.screen, self.animation_offset)
        draw_grid_overlay(self.screen)
        
        # Title with glow effect
        title = FONT_TITLE.render("UNO MODEL BATTLE ARENA", True, COLORS['GOLD'])
        title_glow = FONT_TITLE.render("UNO MODEL BATTLE ARENA", True, (*COLORS['GOLD'][:3], 30))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 40))
        self.screen.blit(title_glow, (title_rect.x + 2, title_rect.y + 2))
        self.screen.blit(title, title_rect)
        
        # Player count selection buttons
        self.player_2_btn.draw(self.screen)
        self.player_3_btn.draw(self.screen)
        self.player_4_btn.draw(self.screen)
        
        player_label = FONT_SMALL.render("Players:", True, COLORS['MUTED'])
        self.screen.blit(player_label, (340, 78))
        
        subtitle = FONT_MEDIUM.render(f"Select {self.num_players} models to compete", True, COLORS['MUTED'])
        sub_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 90))
        self.screen.blit(subtitle, sub_rect)
        
        # Handle hover for active selectors
        mouse_pos = pygame.mouse.get_pos()
        for i in range(self.num_players):
            self.selectors[i].handle_hover(mouse_pos)
        
        # Draw selectors for active players
        for i in range(self.num_players):
            self.selectors[i].draw(self.screen)
        
        # VS badges between selectors
        if self.num_players >= 2:
            for i in range(self.num_players - 1):
                selector = self.selectors[i]
                vs_x = selector.rect.right - 2
                vs_rect = pygame.Rect(vs_x, 270, 40, 28)
                # Glow
                for g in range(2, 0, -1):
                    glow_rect = vs_rect.inflate(g * 3, g * 3)
                    glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, (*COLORS['DANGER'][:3], 25), glow_surf.get_rect(), border_radius=14 + g)
                    self.screen.blit(glow_surf, glow_rect.topleft)
                pygame.draw.rect(self.screen, COLORS['DANGER'], vs_rect, border_radius=14)
                vs_text = FONT_SMALL.render("VS", True, COLORS['WHITE'])
                self.screen.blit(vs_text, (vs_rect.centerx - vs_text.get_width() // 2, 
                                           vs_rect.centery - vs_text.get_height() // 2))
        
        # Buttons
        self.start_button.draw(self.screen)
        self.batch_button.draw(self.screen)
        self.save_button.draw(self.screen)
        self.close_button.draw(self.screen)
        
        # Info panel below buttons
        info_rect = pygame.Rect(25, 455, 380, 230)
        info_surf = pygame.Surface((info_rect.width, info_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(info_surf, (15, 23, 42, 200), info_surf.get_rect(), border_radius=12)
        pygame.draw.rect(info_surf, (30, 41, 59, 80), info_surf.get_rect(), border_radius=12)
        # Top shine
        for y in range(25):
            alpha = int(15 * (1 - y / 25))
            pygame.draw.line(info_surf, (255, 255, 255, alpha), (6, y + 4), (info_rect.width - 6, y + 4))
        self.screen.blit(info_surf, info_rect.topleft)
        pygame.draw.rect(self.screen, COLORS['PANEL_BORDER'], info_rect, 2, border_radius=12)
        
        info_title = FONT_SMALL.render("Available Models", True, COLORS['ACCENT_CYAN'])
        self.screen.blit(info_title, (info_rect.x + 12, info_rect.y + 10))
        
        # Show models in two columns
        y = info_rect.y + 35
        col_width = 180
        for i, model in enumerate(self.models[:14]):
            col = i // 7
            row = i % 7
            if i < 3:
                color = [COLORS['GOLD'], COLORS['SILVER'], COLORS['BRONZE']][i]
            else:
                color = COLORS['LIGHT_GRAY']
            name = model.name[:16] + ".." if len(model.name) > 16 else model.name
            text = FONT_TINY.render(f"{i+1}. {name}", True, color)
            self.screen.blit(text, (info_rect.x + 12 + col * col_width, y + row * 24))
        
        # Instructions on right side
        inst_rect = pygame.Rect(WINDOW_WIDTH - 260, 455, 240, 230)
        inst_surf = pygame.Surface((inst_rect.width, inst_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(inst_surf, (15, 23, 42, 180), inst_surf.get_rect(), border_radius=12)
        self.screen.blit(inst_surf, inst_rect.topleft)
        pygame.draw.rect(self.screen, COLORS['PANEL_BORDER'], inst_rect, 2, border_radius=12)
        
        inst_title = FONT_SMALL.render("Controls", True, COLORS['ACCENT_CYAN'])
        self.screen.blit(inst_title, (inst_rect.x + 12, inst_rect.y + 10))
        
        inst_y = inst_rect.y + 40
        instructions = ["• Click model to select", "• Space = pause/resume", "• ESC = back to menu", "• X = exit"]
        for inst in instructions:
            text = FONT_TINY.render(inst, True, COLORS['MUTED'])
            self.screen.blit(text, (inst_rect.x + 12, inst_y))
            inst_y += 28
    
    def draw_battle_screen(self):
        """Draw battle in progress with modern styling."""
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Animated effects
        draw_ambient_orbs(self.screen, self.animation_offset)
        draw_grid_overlay(self.screen)
        
        # Player colors for up to 4 players
        player_colors = [COLORS['ACCENT_CYAN'], COLORS['ACCENT_PINK'], 
                        COLORS['ACCENT_GREEN'], COLORS['WARNING']]
        
        # === 2 PLAYER MODE: Top vs Bottom layout (like uno_gui) ===
        if self.num_players == 2:
            # Title with score
            title_text = "BATTLE" if not self.batch_mode else f"Batch: {self.batch_games + 1}/{self.batch_target}"
            title = FONT_MEDIUM.render(title_text, True, COLORS['WHITE'])
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 8))
            
            # === PLAYER 1 (TOP - like AI in uno_gui) ===
            p1_panel = pygame.Surface((WINDOW_WIDTH - 60, 100), pygame.SRCALPHA)
            pygame.draw.rect(p1_panel, (0, 0, 0, 40), p1_panel.get_rect(), border_radius=14)
            pygame.draw.rect(p1_panel, (*player_colors[0][:3], 60), p1_panel.get_rect(), 2, border_radius=14)
            self.screen.blit(p1_panel, (30, 35))
            
            hand1 = self.env.hands[0]
            card_spacing = min(50, (WINDOW_WIDTH - 200) // max(len(hand1), 1))
            start_x = (WINDOW_WIDTH - card_spacing * len(hand1)) // 2
            for i, card in enumerate(hand1[:15]):
                card.render(self.screen, start_x + i * card_spacing, 50, scale=0.6)
            
            # P1 badge
            p1_badge = pygame.Surface((180, 28), pygame.SRCALPHA)
            pygame.draw.rect(p1_badge, (*player_colors[0][:3], 200), p1_badge.get_rect(), border_radius=14)
            p1_name = self.battle_models[0].name[:16]
            p1_text = FONT_SMALL.render(f"P1: {p1_name} ({len(hand1)})", True, COLORS['WHITE'])
            p1_badge.blit(p1_text, (10, 5))
            self.screen.blit(p1_badge, (40, 40))
            
            if self.env.current_player == 0:
                turn_ind = FONT_TINY.render("[TURN]", True, COLORS['GOLD'])
                self.screen.blit(turn_ind, (225, 45))
            
            # === CENTER AREA ===
            center_y = WINDOW_HEIGHT // 2 - 50
            
            # Open card with glow
            if self.env.open_card:
                card_color = COLORS.get(self.env.open_card.color, COLORS['WHITE'])
                glow_surf = pygame.Surface((90, 120), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*card_color[:3], 40), glow_surf.get_rect(), border_radius=14)
                self.screen.blit(glow_surf, (WINDOW_WIDTH // 2 - 45, center_y - 5))
                self.env.open_card.render(self.screen, WINDOW_WIDTH // 2 - 32, center_y, scale=0.9)
            
            # Score display
            score_text = FONT_TITLE.render(f"{self.wins[0]} - {self.wins[1]}", True, COLORS['GOLD'])
            self.screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, center_y + 100))
            
            # Direction and turn info
            direction_str = "→→→" if self.env.direction == 1 else "←←←"
            turn_text = f"Turn {self.env.turn_count}  {direction_str}"
            turn_surf = FONT_SMALL.render(turn_text, True, COLORS['MUTED'])
            self.screen.blit(turn_surf, (WINDOW_WIDTH // 2 - turn_surf.get_width() // 2, center_y + 150))
            
            # Last event
            event_surf = FONT_MEDIUM.render(self.last_event, True, COLORS['WARNING'])
            self.screen.blit(event_surf, (WINDOW_WIDTH // 2 - event_surf.get_width() // 2, center_y + 180))
            
            # === PLAYER 2 (BOTTOM - like Player in uno_gui) ===
            p2_panel = pygame.Surface((WINDOW_WIDTH - 80, 130), pygame.SRCALPHA)
            pygame.draw.rect(p2_panel, (0, 0, 0, 50), p2_panel.get_rect(), border_radius=18)
            pygame.draw.rect(p2_panel, (*player_colors[1][:3], 80), p2_panel.get_rect(), 2, border_radius=18)
            self.screen.blit(p2_panel, (40, WINDOW_HEIGHT - 170))
            
            hand2 = self.env.hands[1]
            card_spacing2 = min(60, (WINDOW_WIDTH - 250) // max(len(hand2), 1))
            start_x2 = (WINDOW_WIDTH - card_spacing2 * len(hand2)) // 2
            for i, card in enumerate(hand2[:15]):
                card.render(self.screen, start_x2 + i * card_spacing2, WINDOW_HEIGHT - 155, scale=0.7)
            
            # P2 badge
            p2_badge = pygame.Surface((220, 34), pygame.SRCALPHA)
            pygame.draw.rect(p2_badge, (*player_colors[1][:3], 220), p2_badge.get_rect(), border_radius=17)
            pygame.draw.rect(p2_badge, (255, 255, 255, 40), p2_badge.get_rect(), 1, border_radius=17)
            p2_name = self.battle_models[1].name[:18]
            p2_text = FONT_SMALL.render(f"P2: {p2_name} ({len(hand2)})", True, COLORS['WHITE'])
            p2_badge.blit(p2_text, (12, 8))
            self.screen.blit(p2_badge, (50, WINDOW_HEIGHT - 165))
            
            if self.env.current_player == 1:
                turn_ind = FONT_SMALL.render("● TURN", True, COLORS['GOLD'])
                self.screen.blit(turn_ind, (280, WINDOW_HEIGHT - 158))
            
            # Winner announcement
            if self.game_over and not self.batch_mode:
                winner_name = self.battle_models[self.env.winner].name
                winner_color = player_colors[self.env.winner]
                winner_text = FONT_LARGE.render(f"🏆 P{self.env.winner + 1}: {winner_name[:14]} WINS!", True, COLORS['GOLD'])
                winner_rect = winner_text.get_rect(center=(WINDOW_WIDTH // 2, center_y + 80))
                
                bg_rect = winner_rect.inflate(40, 20)
                pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], bg_rect, border_radius=14)
                pygame.draw.rect(self.screen, winner_color, bg_rect, 3, border_radius=14)
                self.screen.blit(winner_text, winner_rect)
        
        # === 3-4 PLAYER MODE: Four corners layout ===
        else:
            title_text = "BATTLE" if not self.batch_mode else f"Batch: {self.batch_games + 1}/{self.batch_target}"
            title = FONT_LARGE.render(title_text, True, COLORS['WHITE'])
            self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 12))
            
            # Positions: top-left, top-right, bottom-left, bottom-right
            positions = [
                (40, 50, "topleft"),      # P1
                (WINDOW_WIDTH - 340, 50, "topright"),   # P2
                (40, WINDOW_HEIGHT - 160, "bottomleft"),  # P3
                (WINDOW_WIDTH - 340, WINDOW_HEIGHT - 160, "bottomright"),  # P4
            ]
            
            for p in range(self.num_players):
                x, y, pos = positions[p]
                hand = self.env.hands[p]
                color = player_colors[p]
                
                # Panel
                panel = pygame.Surface((300, 110), pygame.SRCALPHA)
                pygame.draw.rect(panel, (0, 0, 0, 50), panel.get_rect(), border_radius=14)
                pygame.draw.rect(panel, (*color[:3], 100), panel.get_rect(), 2, border_radius=14)
                self.screen.blit(panel, (x, y))
                
                # Badge
                name = self.battle_models[p].name[:14]
                badge_text = f"P{p+1}: {name} ({len(hand)})"
                badge = FONT_SMALL.render(badge_text, True, color)
                self.screen.blit(badge, (x + 10, y + 6))
                
                if self.env.current_player == p:
                    turn_ind = FONT_SMALL.render("● TURN", True, COLORS['GOLD'])
                    self.screen.blit(turn_ind, (x + 210, y + 8))
                
                # Cards (compact)
                for i, card in enumerate(hand[:7]):
                    card.render(self.screen, x + 10 + i * 42, y + 30, scale=0.58)
            
            # Center area
            center_y = WINDOW_HEIGHT // 2 - 70
            
            if self.env.open_card:
                card_color = COLORS.get(self.env.open_card.color, COLORS['WHITE'])
                glow_surf = pygame.Surface((90, 115), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (*card_color[:3], 40), glow_surf.get_rect(), border_radius=14)
                self.screen.blit(glow_surf, (WINDOW_WIDTH // 2 - 45, center_y))
                self.env.open_card.render(self.screen, WINDOW_WIDTH // 2 - 38, center_y + 8, scale=0.9)
            
            # Score
            score_parts = " : ".join(str(self.wins[i]) for i in range(self.num_players))
            score_text = FONT_LARGE.render(score_parts, True, COLORS['GOLD'])
            self.screen.blit(score_text, (WINDOW_WIDTH // 2 - score_text.get_width() // 2, center_y + 120))
            
            # Turn info
            direction_str = "→→→" if self.env.direction == 1 else "←←←"
            current_color = player_colors[self.env.current_player]
            turn_text = f"Turn {self.env.turn_count}  •  P{self.env.current_player + 1}  {direction_str}"
            turn_surf = FONT_SMALL.render(turn_text, True, current_color)
            self.screen.blit(turn_surf, (WINDOW_WIDTH // 2 - turn_surf.get_width() // 2, center_y + 155))
            
            # Last event
            event_surf = FONT_MEDIUM.render(self.last_event, True, COLORS['WARNING'])
            self.screen.blit(event_surf, (WINDOW_WIDTH // 2 - event_surf.get_width() // 2, center_y + 185))
            
            # Winner
            if self.game_over and not self.batch_mode:
                winner_name = self.battle_models[self.env.winner].name
                winner_color = player_colors[self.env.winner]
                winner_text = FONT_LARGE.render(f"🏆 P{self.env.winner + 1}: {winner_name[:12]} WINS!", True, COLORS['GOLD'])
                winner_rect = winner_text.get_rect(center=(WINDOW_WIDTH // 2, center_y + 80))
                
                bg_rect = winner_rect.inflate(30, 16)
                pygame.draw.rect(self.screen, COLORS['DARK_GRAY'], bg_rect, border_radius=12)
                pygame.draw.rect(self.screen, winner_color, bg_rect, 3, border_radius=12)
                self.screen.blit(winner_text, winner_rect)
        
        self.back_button.draw(self.screen)
        self.save_button.draw(self.screen)
        self.close_button.draw(self.screen)
    
    def draw_results_screen(self):
        """Draw final results with modern styling."""
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Animated effects
        draw_ambient_orbs(self.screen, self.animation_offset)
        draw_grid_overlay(self.screen)
        
        # Title with glow
        title = FONT_TITLE.render("BATTLE RESULTS", True, COLORS['GOLD'])
        title_glow = FONT_TITLE.render("BATTLE RESULTS", True, (*COLORS['GOLD'][:3], 30))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 35))
        self.screen.blit(title_glow, (title_rect.x + 2, title_rect.y + 2))
        self.screen.blit(title, title_rect)
        
        # Models vs text
        if self.num_players == 2:
            vs_text = FONT_MEDIUM.render(f"{self.battle_models[0].name[:14]} VS {self.battle_models[1].name[:14]}", 
                                         True, COLORS['WHITE'])
        else:
            names = " vs ".join(m.name[:10] for m in self.battle_models)
            vs_text = FONT_SMALL.render(names, True, COLORS['WHITE'])
        vs_rect = vs_text.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(vs_text, vs_rect)
        
        # Big score
        total = sum(self.wins[:self.num_players])
        player_colors = [COLORS['ACCENT_CYAN'], COLORS['ACCENT_PINK'], 
                        COLORS['ACCENT_GREEN'], COLORS['WARNING']]
        
        if self.num_players == 2:
            score_str = f"{self.wins[0]} - {self.wins[1]}"
            score = FONT_TITLE.render(score_str, True, COLORS['GOLD'])
        else:
            score_str = " : ".join(str(self.wins[i]) for i in range(self.num_players))
            score = FONT_LARGE.render(score_str, True, COLORS['GOLD'])
        score_rect = score.get_rect(center=(WINDOW_WIDTH // 2, 125))
        self.screen.blit(score, score_rect)
        
        # Percentages
        pct_y = 165
        pct_x_start = WINDOW_WIDTH // 2 - (self.num_players * 60) // 2
        for i in range(self.num_players):
            pct = (self.wins[i] / total * 100) if total > 0 else 0
            pct_text = FONT_TINY.render(f"P{i+1}: {pct:.1f}%", True, player_colors[i])
            self.screen.blit(pct_text, (pct_x_start + i * 80, pct_y))
        
        # Winner banner
        max_wins = max(self.wins[:self.num_players])
        winners = [i for i in range(self.num_players) if self.wins[i] == max_wins]
        
        if len(winners) == 1:
            winner_idx = winners[0]
            winner = f"** P{winner_idx + 1}: {self.battle_models[winner_idx].name[:14]} WINS! **"
            color = player_colors[winner_idx]
        else:
            winner = "-- TIE! --"
            color = COLORS['WARNING']
        
        winner_surf = FONT_LARGE.render(winner, True, color)
        winner_rect = winner_surf.get_rect(center=(WINDOW_WIDTH // 2, 210))
        
        for i in range(3, 0, -1):
            glow_rect = winner_rect.inflate(i * 8, i * 4)
            glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*color[:3], 15), glow_surf.get_rect(), border_radius=10 + i * 2)
            self.screen.blit(glow_surf, glow_rect.topleft)
        
        self.screen.blit(winner_surf, winner_rect)
        
        # Stats panel
        stats_rect = pygame.Rect(120, 255, WINDOW_WIDTH - 240, 360)
        stats_surf = pygame.Surface((stats_rect.width, stats_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(stats_surf, (15, 23, 42, 200), stats_surf.get_rect(), border_radius=14)
        for y in range(30):
            alpha = int(15 * (1 - y / 30))
            pygame.draw.line(stats_surf, (255, 255, 255, alpha), (8, y + 4), (stats_rect.width - 8, y + 4))
        self.screen.blit(stats_surf, stats_rect.topleft)
        pygame.draw.rect(self.screen, COLORS['PANEL_BORDER'], stats_rect, 2, border_radius=14)
        
        stats_title = FONT_MEDIUM.render("Statistics", True, COLORS['ACCENT_PURPLE'])
        self.screen.blit(stats_title, (stats_rect.x + 20, stats_rect.y + 15))
        
        # Calculate stats
        if self.results:
            avg_turns = sum(r['turns'] for r in self.results) / len(self.results)
            
            stats = [
                f"Total Games: {len(self.results)}",
                f"Players: {self.num_players}",
                f"Average Game Length: {avg_turns:.1f} turns",
            ]
            
            # Add per-player stats
            for i in range(self.num_players):
                pct = (self.wins[i] / total * 100) if total > 0 else 0
                cards_key = f"model{i+1}_cards"
                if cards_key in self.results[0]:
                    avg_cards = sum(r.get(cards_key, 0) for r in self.results) / len(self.results)
                    stats.append(f"P{i+1} ({self.battle_models[i].name[:12]}): {self.wins[i]} wins ({pct:.1f}%), avg {avg_cards:.1f} cards")
            
            for i, stat in enumerate(stats):
                stat_surf = FONT_SMALL.render(stat, True, COLORS['WHITE'])
                self.screen.blit(stat_surf, (stats_rect.x + 30, stats_rect.y + 55 + i * 28))
        
        # Buttons
        self.back_button.draw(self.screen)
        self.save_button.draw(self.screen)
        self.close_button.draw(self.screen)
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Close button works on all screens
                        if self.close_button.clicked(mouse_pos):
                            running = False
                            continue
                        
                        if self.state == "SELECT":
                            # Player count buttons
                            if self.player_2_btn.clicked(mouse_pos):
                                self.set_num_players(2)
                            elif self.player_3_btn.clicked(mouse_pos):
                                self.set_num_players(3)
                            elif self.player_4_btn.clicked(mouse_pos):
                                self.set_num_players(4)
                            
                            # Handle selector clicks for active players
                            for i in range(self.num_players):
                                self.selectors[i].handle_click(mouse_pos)
                            
                            if self.start_button.clicked(mouse_pos):
                                self.start_battle(batch=False)
                            if self.batch_button.clicked(mouse_pos):
                                self.start_battle(batch=True)
                        
                        elif self.state == "BATTLE":
                            if self.back_button.clicked(mouse_pos):
                                self.state = "SELECT"
                        
                        elif self.state == "RESULTS":
                            if self.back_button.clicked(mouse_pos):
                                self.state = "SELECT"
                            if self.save_button.clicked(mouse_pos):
                                saved = self.save_results()
                                if saved:
                                    self.last_event = f"Saved to {saved}"
                    
                    elif event.button == 4:  # Scroll up
                        if self.state == "SELECT":
                            for i in range(self.num_players):
                                if self.selectors[i].rect.collidepoint(mouse_pos):
                                    self.selectors[i].handle_scroll(-1)
                    elif event.button == 5:  # Scroll down
                        if self.state == "SELECT":
                            for i in range(self.num_players):
                                if self.selectors[i].rect.collidepoint(mouse_pos):
                                    self.selectors[i].handle_scroll(1)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state != "SELECT":
                            self.state = "SELECT"
                        else:
                            running = False
                    if event.key == pygame.K_SPACE and self.state == "BATTLE":
                        self.auto_play = not self.auto_play
            
            # Update button hover states
            self.close_button.update(mouse_pos)
            
            if self.state == "SELECT":
                self.start_button.update(mouse_pos)
                self.batch_button.update(mouse_pos)
                self.player_2_btn.update(mouse_pos)
                self.player_3_btn.update(mouse_pos)
                self.player_4_btn.update(mouse_pos)
            
            elif self.state == "BATTLE":
                self.back_button.update(mouse_pos)
                
                # Auto-play turns
                if self.auto_play and not self.game_over:
                    self.frame_count += 1
                    speed = 1 if self.batch_mode else self.battle_speed
                    if self.frame_count >= speed:
                        self.run_turn()
                        self.frame_count = 0
            
            elif self.state == "RESULTS":
                self.back_button.update(mouse_pos)
                self.save_button.update(mouse_pos)
            
            # Draw
            self.animation_offset += 1  # Animate background effects
            
            if self.state == "SELECT":
                self.draw_select_screen()
            elif self.state == "BATTLE":
                self.draw_battle_screen()
            elif self.state == "RESULTS":
                self.draw_results_screen()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    print("=" * 50)
    print("   UNO Model Battle Arena")
    print("=" * 50)
    
    gui = ModelBattleGUI()
    gui.run()
