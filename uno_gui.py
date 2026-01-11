import pygame
import sys
import random
import math
import os
from enum import Enum
from typing import List, Optional, Tuple, Dict

# Try to import RL agent components
try:
    import pandas as pd
    import numpy as np
    from src.agents import QLearningAgent
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: Legacy RL agent not available.")

# Try to import Stable Baselines3 agent (preferred)
try:
    from stable_baselines3 import PPO, DQN
    import torch
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable Baselines3 not available. Install with: pip install stable-baselines3")

# Try to import RecurrentPPO for LSTM-based models
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False
    print("Warning: RecurrentPPO not available. Install with: pip install sb3-contrib")

# Initialize Pygame
pygame.init()
pygame.font.init()

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Window settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750
FPS = 60

# Premium color palette - ultra-modern and elegant
COLORS = {
    # Card colors - vibrant and eye-catching
    'RED': (239, 68, 68),
    'GRE': (16, 185, 129),
    'BLU': (59, 130, 246),
    'YEL': (250, 204, 21),
    'WILD': (24, 24, 36),
    
    # UI colors
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (156, 163, 175),
    'LIGHT_GRAY': (243, 244, 246),
    'DARK_GRAY': (45, 55, 72),
    'MUTED': (148, 163, 184),
    
    # Background gradient colors - deep space theme
    'BG_TOP': (5, 8, 22),
    'BG_BOTTOM': (15, 23, 42),
    'BG_ACCENT': (30, 41, 59),
    'BG_GLOW': (99, 102, 241),
    
    # Accent colors - neon modern palette
    'GOLD': (251, 191, 36),
    'ACCENT_BLUE': (56, 189, 248),
    'ACCENT_PURPLE': (192, 132, 252),
    'ACCENT_CYAN': (34, 211, 238),
    'ACCENT_GREEN': (74, 222, 128),
    'ACCENT_ROSE': (251, 113, 133),
    'ACCENT_ORANGE': (251, 146, 60),
    'ACCENT_INDIGO': (165, 180, 252),
    'ACCENT_PINK': (244, 114, 182),
    'ACCENT_EMERALD': (52, 211, 153),
    'ACCENT_VIOLET': (167, 139, 250),
    
    # UI element colors - glassmorphism 2.0
    'PANEL_BG': (15, 23, 42, 200),
    'PANEL_BORDER': (51, 65, 85),
    'GLASS_WHITE': (255, 255, 255, 8),
    'GLASS_BORDER': (255, 255, 255, 20),
    'GLASS_HIGHLIGHT': (255, 255, 255, 40),
    
    # Button colors - vibrant gradients
    'BUTTON_PRIMARY': (79, 70, 229),
    'BUTTON_PRIMARY_HOVER': (99, 102, 241),
    'BUTTON_SUCCESS': (16, 185, 129),
    'BUTTON_SUCCESS_HOVER': (52, 211, 153),
    'BUTTON_DANGER': (239, 68, 68),
    'BUTTON_DANGER_HOVER': (248, 113, 113),
    'BUTTON_WARNING': (245, 158, 11),
    
    # Effects - enhanced glow
    'GLOW_WHITE': (255, 255, 255, 120),
    'GLOW_PURPLE': (192, 132, 252, 80),
    'GLOW_BLUE': (56, 189, 248, 80),
    'GLOW_CYAN': (34, 211, 238, 80),
    'SHADOW': (0, 0, 0, 120),
    'SHADOW_SOFT': (0, 0, 0, 50),
}

# Card dimensions - optimized for modern displays
CARD_WIDTH = 88
CARD_HEIGHT = 125
CARD_RADIUS = 16

# Premium typography - Inter-like modern font system
try:
    # Try Segoe UI for modern Windows look, with better sizes
    FONT_BOLD_LARGE = pygame.font.SysFont('segoeui', 52, bold=True)
    FONT_BOLD_MEDIUM = pygame.font.SysFont('segoeui', 36, bold=True)
    FONT_REGULAR = pygame.font.SysFont('segoeui', 26)
    FONT_SMALL = pygame.font.SysFont('segoeui', 22)
    FONT_TINY = pygame.font.SysFont('segoeui', 17)
    FONT_MICRO = pygame.font.SysFont('segoeui', 14)
    FONT_CARD = pygame.font.SysFont('segoeui', 30, bold=True)
    FONT_CARD_LARGE = pygame.font.SysFont('segoeui', 44, bold=True)
    FONT_TITLE = pygame.font.SysFont('segoeui', 80, bold=True)
    FONT_SUBTITLE = pygame.font.SysFont('segoeui', 18)
    FONT_HEADER = pygame.font.SysFont('segoeui', 28, bold=True)
except:
    FONT_BOLD_LARGE = pygame.font.Font(None, 56)
    FONT_BOLD_MEDIUM = pygame.font.Font(None, 40)
    FONT_REGULAR = pygame.font.Font(None, 28)
    FONT_SMALL = pygame.font.Font(None, 24)
    FONT_TINY = pygame.font.Font(None, 18)
    FONT_MICRO = pygame.font.Font(None, 15)
    FONT_CARD = pygame.font.Font(None, 34)
    FONT_CARD_LARGE = pygame.font.Font(None, 48)
    FONT_TITLE = pygame.font.Font(None, 84)
    FONT_SUBTITLE = pygame.font.Font(None, 20)
    FONT_HEADER = pygame.font.Font(None, 32)

# Legacy font references for compatibility
FONT_LARGE = FONT_BOLD_LARGE
FONT_MEDIUM = FONT_BOLD_MEDIUM


# =============================================================================
# MODEL WRAPPER FOR DIFFERENT AI TYPES
# =============================================================================

class ModelInfo:
    """Information about an available AI model."""
    
    def __init__(self, name: str, path: str, algo: str, win_rate: str = ""):
        self.name = name
        self.path = path
        self.algo = algo  # 'ppo', 'dqn', 'recurrentppo', 'random'
        self.win_rate = win_rate
        self.model = None
        self.is_recurrent = algo == 'recurrentppo'
        self.lstm_states = None
    
    def load(self) -> bool:
        """Load the model from disk."""
        if self.algo == 'random':
            return True
        
        try:
            if self.algo == 'recurrentppo' and RECURRENT_PPO_AVAILABLE:
                self.model = RecurrentPPO.load(self.path)
                return True
            elif self.algo == 'ppo' and SB3_AVAILABLE:
                self.model = PPO.load(self.path)
                return True
            elif self.algo == 'dqn' and SB3_AVAILABLE:
                self.model = DQN.load(self.path)
                return True
        except Exception as e:
            print(f"Failed to load {self.name}: {e}")
        return False


def discover_models() -> List[ModelInfo]:
    """Discover all available AI models."""
    models = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    # Always add random AI option
    models.append(ModelInfo("Random AI", "", "random", "~25%"))
    
    # Known model configurations with win rates
    known_models = {
        "selfplay_champion.zip": ("Self-Play Champion", "recurrentppo", "70%+"),
        "best_recurrent_ppo_uno.zip": ("Best Recurrent PPO", "recurrentppo", "60%"),
        "optimal_recurrent_ppo.zip": ("Optimal Recurrent PPO", "recurrentppo", "59%"),
        "sb3_recurrentppo_uno.zip": ("SB3 Recurrent PPO", "recurrentppo", "57%"),
        "recurrent_ppo_uno.zip": ("Recurrent PPO", "recurrentppo", "55%"),
        "sb3_ppo_uno.zip": ("PPO Standard", "ppo", "45%"),
        "best_ppo_uno.zip": ("Best PPO", "ppo", "48%"),
        "sb3_dqn_uno.zip": ("DQN", "dqn", "40%"),
        "best_model.zip": ("Best Model", "ppo", ""),
    }
    
    if os.path.exists(models_dir):
        # First add known models in order
        for filename, (name, algo, win_rate) in known_models.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                models.append(ModelInfo(name, path, algo, win_rate))
        
        # Then discover any other .zip files
        for filename in os.listdir(models_dir):
            if filename.endswith('.zip') and filename not in known_models:
                path = os.path.join(models_dir, filename)
                # Try to infer algorithm from filename
                if 'recurrent' in filename.lower() or 'rppo' in filename.lower():
                    algo = 'recurrentppo'
                elif 'dqn' in filename.lower():
                    algo = 'dqn'
                else:
                    algo = 'ppo'
                name = filename.replace('.zip', '').replace('_', ' ').title()
                models.append(ModelInfo(name, path, algo))
    
    return models


class ModelSelector:
    """Dropdown-style model selector widget."""
    
    def __init__(self, x: int, y: int, width: int, height: int, title: str):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.models = discover_models()
        self.selected_idx = 0
        self.expanded = False
        self.scroll_offset = 0
        self.item_height = 36
        self.hover_idx = -1
        self.accent_color = COLORS['ACCENT_CYAN']
        
        # Pre-select best model if available (Self-Play Champion is the best)
        for i, m in enumerate(self.models):
            if 'Self-Play Champion' in m.name:
                self.selected_idx = i
                break
            elif 'Best Recurrent' in m.name and self.selected_idx == 0:
                self.selected_idx = i
    
    @property
    def collapsed_height(self) -> int:
        return 48
    
    @property
    def expanded_height(self) -> int:
        return min(48 + len(self.models) * self.item_height + 10, 300)
    
    def draw(self, screen):
        """Draw the selector."""
        current_height = self.expanded_height if self.expanded else self.collapsed_height
        draw_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, current_height)
        
        # Glassmorphism panel
        panel_surf = pygame.Surface((draw_rect.width, draw_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (15, 23, 42, 220), panel_surf.get_rect(), border_radius=12)
        pygame.draw.rect(panel_surf, (30, 41, 59, 100), panel_surf.get_rect(), border_radius=12)
        screen.blit(panel_surf, draw_rect.topleft)
        
        # Border with glow
        for i in range(2, 0, -1):
            border_rect = draw_rect.inflate(i * 2, i * 2)
            alpha = int(40 / i)
            pygame.draw.rect(screen, (*self.accent_color[:3], alpha), border_rect, 1, border_radius=12 + i)
        pygame.draw.rect(screen, self.accent_color, draw_rect, 2, border_radius=12)
        
        # Title and selected model
        title_text = FONT_TINY.render(self.title, True, self.accent_color)
        screen.blit(title_text, (self.rect.x + 12, self.rect.y + 4))
        
        selected = self.models[self.selected_idx] if self.models else None
        if selected:
            name = selected.name
            if selected.win_rate:
                name += f" ({selected.win_rate})"
            selected_text = FONT_SMALL.render(name, True, COLORS['WHITE'])
            screen.blit(selected_text, (self.rect.x + 12, self.rect.y + 22))
        
        # Dropdown arrow
        arrow = "▼" if not self.expanded else "▲"
        arrow_text = FONT_SMALL.render(arrow, True, COLORS['LIGHT_GRAY'])
        screen.blit(arrow_text, (self.rect.x + self.rect.width - 30, self.rect.y + 20))
        
        # Expanded list
        if self.expanded:
            list_y = self.rect.y + 50
            visible_items = (current_height - 60) // self.item_height
            
            for i in range(min(visible_items, len(self.models))):
                idx = i + self.scroll_offset
                if idx >= len(self.models):
                    break
                
                model = self.models[idx]
                item_rect = pygame.Rect(self.rect.x + 6, list_y + i * self.item_height,
                                       self.rect.width - 12, self.item_height - 4)
                
                item_surf = pygame.Surface((item_rect.width, item_rect.height), pygame.SRCALPHA)
                
                if idx == self.selected_idx:
                    # Selected item
                    pygame.draw.rect(item_surf, (*self.accent_color, 180), item_surf.get_rect(), border_radius=8)
                    text_color = COLORS['WHITE']
                elif idx == self.hover_idx:
                    # Hover item
                    pygame.draw.rect(item_surf, (51, 65, 85, 200), item_surf.get_rect(), border_radius=8)
                    text_color = COLORS['WHITE']
                else:
                    # Normal item
                    pygame.draw.rect(item_surf, (30, 41, 59, 150), item_surf.get_rect(), border_radius=8)
                    text_color = COLORS['LIGHT_GRAY']
                
                screen.blit(item_surf, item_rect.topleft)
                
                # Model name with win rate
                display_name = model.name
                if model.win_rate:
                    display_name += f" ({model.win_rate})"
                if len(display_name) > 30:
                    display_name = display_name[:27] + "..."
                text = FONT_TINY.render(display_name, True, text_color)
                screen.blit(text, (item_rect.x + 10, item_rect.y + 8))
    
    def handle_event(self, event) -> bool:
        """Handle events. Returns True if model changed."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # Check header click (toggle expand)
                header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 48)
                if header_rect.collidepoint(event.pos):
                    self.expanded = not self.expanded
                    return False
                
                # Check item click when expanded
                if self.expanded:
                    list_y = self.rect.y + 50
                    current_height = self.expanded_height
                    visible_items = (current_height - 60) // self.item_height
                    
                    for i in range(min(visible_items, len(self.models))):
                        idx = i + self.scroll_offset
                        if idx >= len(self.models):
                            break
                        item_rect = pygame.Rect(self.rect.x + 6, list_y + i * self.item_height,
                                               self.rect.width - 12, self.item_height - 4)
                        if item_rect.collidepoint(event.pos):
                            if self.selected_idx != idx:
                                self.selected_idx = idx
                                self.expanded = False
                                return True  # Model changed
                            self.expanded = False
                            return False
                
                # Click outside - collapse
                if self.expanded:
                    draw_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, self.expanded_height)
                    if not draw_rect.collidepoint(event.pos):
                        self.expanded = False
            
            elif event.button == 4:  # Scroll up
                if self.expanded:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
            elif event.button == 5:  # Scroll down
                if self.expanded:
                    max_scroll = max(0, len(self.models) - 5)
                    self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
        
        elif event.type == pygame.MOUSEMOTION:
            if self.expanded:
                list_y = self.rect.y + 50
                current_height = self.expanded_height
                visible_items = (current_height - 60) // self.item_height
                
                self.hover_idx = -1
                for i in range(min(visible_items, len(self.models))):
                    idx = i + self.scroll_offset
                    item_rect = pygame.Rect(self.rect.x + 6, list_y + i * self.item_height,
                                           self.rect.width - 12, self.item_height - 4)
                    if item_rect.collidepoint(event.pos):
                        self.hover_idx = idx
                        break
        
        return False
    
    def get_selected(self) -> Optional[ModelInfo]:
        """Get the currently selected model."""
        if 0 <= self.selected_idx < len(self.models):
            return self.models[self.selected_idx]
        return None


# =============================================================================
# CARD CLASS (Adapted from project)
# =============================================================================

class Card:
    """Card representation with graphical rendering capabilities."""
    
    def __init__(self, color: str, value):
        self.color = color
        self.value = value
        self.rect = pygame.Rect(0, 0, CARD_WIDTH, CARD_HEIGHT)
        self.target_pos = (0, 0)
        self.hover = False
        self.selected = False
        self.angle = 0
        
    def evaluate_card(self, open_color: str, open_value) -> bool:
        """Check if this card can be played on the open card."""
        if self.color == open_color:
            return True
        if self.value == open_value:
            return True
        if self.value in ["COL", "PL4"]:
            return True
        return False
    
    def get_display_value(self) -> str:
        """Get the display string for the card value."""
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
    
    def get_color_rgb(self) -> Tuple[int, int, int]:
        """Get the RGB color for this card."""
        if self.color in COLORS:
            return COLORS[self.color]
        return COLORS['WILD']
    
    def render(self, surface: pygame.Surface, x: int, y: int, 
               face_up: bool = True, scale: float = 1.0):
        """Render the card at the specified position with premium styling."""
        self.rect.x = x
        self.rect.y = y
        
        w = int(CARD_WIDTH * scale)
        h = int(CARD_HEIGHT * scale)
        
        # Create card surface with extra space for effects
        margin = 12
        card_surface = pygame.Surface((w + margin * 2, h + margin * 2), pygame.SRCALPHA)
        
        # Draw layered soft shadow for depth
        for i in range(4):
            shadow_alpha = 35 - i * 8
            shadow_offset = 4 + i
            shadow_rect = pygame.Rect(margin + i, margin + shadow_offset, w, h)
            pygame.draw.rect(card_surface, (0, 0, 0, shadow_alpha), shadow_rect, 
                           border_radius=int(CARD_RADIUS * scale))
        
        # Draw card background
        card_rect = pygame.Rect(margin, margin, w, h)
        
        if face_up:
            base_color = self.get_color_rgb()
            
            # Draw main card body
            pygame.draw.rect(card_surface, base_color, card_rect, 
                           border_radius=int(CARD_RADIUS * scale))
            
            # Top gradient highlight for 3D effect
            highlight_h = int(h * 0.35)
            highlight_surf = pygame.Surface((w - 6, highlight_h), pygame.SRCALPHA)
            for hy in range(highlight_h):
                alpha = int(45 * (1 - hy / highlight_h))
                pygame.draw.line(highlight_surf, (255, 255, 255, alpha), (0, hy), (w - 6, hy))
            card_surface.blit(highlight_surf, (margin + 3, margin + 2))
            
            # Draw inner white oval with subtle rotation
            inner_margin = int(8 * scale)
            oval_surface = pygame.Surface((w, h), pygame.SRCALPHA)
            oval_rect = pygame.Rect(inner_margin, int(20 * scale), 
                                   w - 2 * inner_margin, h - int(40 * scale))
            pygame.draw.ellipse(oval_surface, (255, 255, 255, 240), oval_rect)
            
            # Subtle inner shadow on oval
            inner_shadow = pygame.Rect(inner_margin + 2, int(22 * scale), 
                                      w - 2 * inner_margin - 4, h - int(44 * scale))
            pygame.draw.ellipse(oval_surface, (0, 0, 0, 15), inner_shadow, width=2)
            
            # Rotate the oval for classic UNO style
            rotated_oval = pygame.transform.rotate(oval_surface, 28)
            oval_pos = (margin + w // 2 - rotated_oval.get_width() // 2,
                       margin + h // 2 - rotated_oval.get_height() // 2)
            card_surface.blit(rotated_oval, oval_pos)
            
            # Draw value with shadow for depth
            display_val = self.get_display_value()
            font = FONT_CARD_LARGE if scale >= 0.9 else FONT_CARD
            
            # Text shadow layers
            for offset in [(2, 2), (1, 1)]:
                text_shadow = font.render(display_val, True, (0, 0, 0, 100))
                shadow_rect = text_shadow.get_rect(center=(margin + w // 2 + offset[0], 
                                                          margin + h // 2 + offset[1]))
                card_surface.blit(text_shadow, shadow_rect)
            
            # Main value text
            text_main = font.render(display_val, True, base_color)
            text_rect = text_main.get_rect(center=(margin + w // 2, margin + h // 2))
            card_surface.blit(text_main, text_rect)
            
            # Draw corner values with subtle styling
            corner_font = FONT_TINY
            corner_text = corner_font.render(display_val, True, COLORS['WHITE'])
            
            # Top-left corner with shadow
            corner_shadow = corner_font.render(display_val, True, (0, 0, 0, 100))
            card_surface.blit(corner_shadow, (margin + int(9 * scale), margin + int(9 * scale)))
            card_surface.blit(corner_text, (margin + int(8 * scale), margin + int(8 * scale)))
            
            # Bottom right corner (rotated 180)
            rotated_corner = pygame.transform.rotate(corner_text, 180)
            card_surface.blit(rotated_corner, (margin + w - int(20 * scale), margin + h - int(22 * scale)))
            
            # Hover/selection glow effect
            if self.hover or self.selected:
                glow_color = (255, 255, 255) if self.hover else (255, 215, 0)
                glow_intensity = 4 if self.hover else 3
                for i in range(glow_intensity):
                    glow_alpha = 80 - i * 18
                    glow_rect = pygame.Rect(margin - i - 1, margin - i - 1, 
                                          w + (i + 1) * 2, h + (i + 1) * 2)
                    pygame.draw.rect(card_surface, (*glow_color, glow_alpha), glow_rect,
                                   width=2, border_radius=int(CARD_RADIUS * scale) + i + 1)
            
        else:
            # Draw card back with premium design
            pygame.draw.rect(card_surface, COLORS['WILD'], card_rect, 
                           border_radius=int(CARD_RADIUS * scale))
            
            # Inner border with gradient feel
            inner_border = pygame.Rect(margin + 4, margin + 4, w - 8, h - 8)
            pygame.draw.rect(card_surface, (50, 50, 70), inner_border, 
                           width=2, border_radius=int(CARD_RADIUS * scale) - 3)
            
            # Draw UNO oval on back
            oval_rect = pygame.Rect(margin + int(10 * scale), margin + int(18 * scale), 
                                   w - int(20 * scale), h - int(36 * scale))
            pygame.draw.ellipse(card_surface, COLORS['RED'], oval_rect)
            
            # Inner yellow oval
            inner_oval = pygame.Rect(margin + int(14 * scale), margin + int(26 * scale), 
                                    w - int(28 * scale), h - int(52 * scale))
            pygame.draw.ellipse(card_surface, COLORS['YEL'], inner_oval)
            
            # UNO text with shadow
            uno_font = FONT_CARD if scale >= 0.9 else FONT_TINY
            uno_shadow = uno_font.render("UNO", True, (0, 0, 0, 100))
            uno_text = uno_font.render("UNO", True, COLORS['RED'])
            uno_rotated_shadow = pygame.transform.rotate(uno_shadow, -28)
            uno_rotated = pygame.transform.rotate(uno_text, -28)
            text_rect = uno_rotated.get_rect(center=(margin + w // 2, margin + h // 2))
            card_surface.blit(uno_rotated_shadow, (text_rect.x + 1, text_rect.y + 1))
            card_surface.blit(uno_rotated, text_rect)
        
        surface.blit(card_surface, (x - margin, y - margin))
        return pygame.Rect(x, y, w, h)


# =============================================================================
# DECK CLASS (Adapted from project)
# =============================================================================

class Deck:
    """Deck of UNO cards."""
    
    def __init__(self):
        self.cards: List[Card] = []
        self.cards_disc: List[Card] = []
        self.build()
        self.shuffle()
    
    def build(self):
        """Build a standard UNO deck."""
        colors = ["RED", "GRE", "BLU", "YEL"]
        
        # Zero cards (one of each color)
        for c in colors:
            self.cards.append(Card(c, 0))
        
        # Number cards 1-9 (two of each)
        for _ in range(2):
            for c in colors:
                for v in range(1, 10):
                    self.cards.append(Card(c, v))
        
        # Action cards (two of each)
        for _ in range(2):
            for c in colors:
                for v in ["SKI", "REV", "PL2"]:
                    self.cards.append(Card(c, v))
        
        # Wild cards (four of each)
        for _ in range(4):
            self.cards.append(Card("WILD", "COL"))
            self.cards.append(Card("WILD", "PL4"))
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def draw(self) -> Optional[Card]:
        """Draw a card from the deck."""
        if len(self.cards) == 0:
            if len(self.cards_disc) == 0:
                return None
            self.cards = self.cards_disc[:-1]  # Keep the top card
            self.cards_disc = [self.cards_disc[-1]] if self.cards_disc else []
            self.shuffle()
        
        if self.cards:
            return self.cards.pop()
        return None
    
    def discard(self, card: Card):
        """Add a card to the discard pile."""
        self.cards_disc.append(card)


# =============================================================================
# GAME STATE
# =============================================================================

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    PLAYER_TURN = "player_turn"
    AI_TURN = "ai_turn"
    RL_PLAYER_TURN = "rl_player_turn"
    CHOOSING_COLOR = "choosing_color"
    GAME_OVER = "game_over"
    DRAWING = "drawing"


# =============================================================================
# BUTTON CLASS
# =============================================================================

class Button:
    """Premium interactive button with glassmorphism effects and smooth animations."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, color: Tuple[int, int, int], 
                 hover_color: Optional[Tuple[int, int, int]] = None,
                 icon: str = None, style: str = "filled"):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.original_color = color
        self.hover_color = hover_color or tuple(min(c + 40, 255) for c in color)
        self.is_hovered = False
        self.icon = icon
        self.style = style  # 'filled', 'outline', 'glass', 'gradient'
        self.hover_scale = 1.0
        self.target_scale = 1.0
        self.glow_intensity = 0.0
        self.press_offset = 0
        self.ripple_alpha = 0
        self.border_glow = 0
        
    def update(self, mouse_pos: Tuple[int, int]):
        """Update button hover state with smooth animation."""
        was_hovered = self.is_hovered
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        self.target_scale = 1.03 if self.is_hovered else 1.0
        # Smooth scale transition
        self.hover_scale += (self.target_scale - self.hover_scale) * 0.15
        # Glow animation
        target_glow = 1.0 if self.is_hovered else 0.0
        self.glow_intensity += (target_glow - self.glow_intensity) * 0.2
    
    def render(self, surface: pygame.Surface):
        """Render the button with ultra-modern styling."""
        # Calculate scaled dimensions
        scale = self.hover_scale
        w = int(self.rect.width * scale)
        h = int(self.rect.height * scale)
        x = self.rect.x - (w - self.rect.width) // 2
        y = self.rect.y - (h - self.rect.height) // 2 + self.press_offset
        
        # Create button surface with extra space for effects
        margin = 24
        btn_surface = pygame.Surface((w + margin * 2, h + margin * 2), pygame.SRCALPHA)
        
        # Enhanced glow effect with color bleed
        if self.glow_intensity > 0.05:
            glow_alpha = int(60 * self.glow_intensity)
            for i in range(5):
                glow_size = 12 - i * 2
                glow_color = (*self.color[:3], glow_alpha // (i + 1))
                glow_rect = pygame.Rect(margin - glow_size, margin - glow_size, 
                                       w + glow_size * 2, h + glow_size * 2)
                pygame.draw.rect(btn_surface, glow_color, glow_rect, 
                               border_radius=16 + glow_size)
        
        # Layered soft shadow for depth
        for i in range(3):
            shadow_offset = 3 + i * 2 if not self.is_hovered else 5 + i * 2
            shadow_alpha = (60 - i * 15) if self.is_hovered else (45 - i * 12)
            shadow_rect = pygame.Rect(margin + 1, margin + shadow_offset, w, h)
            pygame.draw.rect(btn_surface, (0, 0, 0, shadow_alpha), shadow_rect, border_radius=14)
        
        # Draw button background
        btn_rect = pygame.Rect(margin, margin, w, h)
        
        if self.style == 'glass':
            # Modern glassmorphism style
            pygame.draw.rect(btn_surface, (255, 255, 255, 12), btn_rect, border_radius=14)
            # Top edge highlight
            top_highlight = pygame.Rect(margin + 1, margin + 1, w - 2, 1)
            pygame.draw.rect(btn_surface, (255, 255, 255, 50), top_highlight, border_radius=14)
            pygame.draw.rect(btn_surface, (255, 255, 255, 30), btn_rect, width=1, border_radius=14)
        elif self.style == 'outline':
            # Outline style with inner glow
            pygame.draw.rect(btn_surface, (*self.color, 20), btn_rect, border_radius=14)
            border_color = self.hover_color if self.is_hovered else self.color
            pygame.draw.rect(btn_surface, border_color, btn_rect, width=2, border_radius=14)
        elif self.style == 'gradient':
            # Gradient button effect
            for gy in range(h):
                progress = gy / h
                r = int(self.color[0] * (1 - progress * 0.3))
                g = int(self.color[1] * (1 - progress * 0.3))
                b = int(self.color[2] * (1 - progress * 0.3))
                pygame.draw.line(btn_surface, (r, g, b), 
                               (margin + 3, margin + gy), (margin + w - 3, margin + gy))
            pygame.draw.rect(btn_surface, (0, 0, 0, 0), btn_rect, border_radius=14)
        else:
            # Filled style with enhanced gradient
            color = self.hover_color if self.is_hovered else self.color
            pygame.draw.rect(btn_surface, color, btn_rect, border_radius=14)
            
            # Top highlight for premium 3D effect
            highlight_h = h // 2
            highlight_surf = pygame.Surface((w - 4, highlight_h), pygame.SRCALPHA)
            for hy in range(highlight_h):
                alpha = int(45 * (1 - hy / highlight_h) ** 1.5)
                pygame.draw.line(highlight_surf, (255, 255, 255, alpha), (0, hy), (w - 4, hy))
            btn_surface.blit(highlight_surf, (margin + 2, margin + 2))
            
            # Inner border highlight
            pygame.draw.rect(btn_surface, (255, 255, 255, 25), btn_rect, width=1, border_radius=14)
            
            # Subtle bottom darkening for depth
            bottom_h = 6
            bottom_surf = pygame.Surface((w - 4, bottom_h), pygame.SRCALPHA)
            for by in range(bottom_h):
                alpha = int(25 * (by / bottom_h))
                pygame.draw.line(bottom_surf, (0, 0, 0, alpha), (0, by), (w - 4, by))
            btn_surface.blit(bottom_surf, (margin + 2, margin + h - bottom_h - 1))
        
        # Render text with better shadow
        text_color = COLORS['WHITE']
        # Multi-layer text shadow for depth
        for offset in [(2, 2), (1, 1)]:
            shadow_alpha = 60 if offset == (2, 2) else 40
            text_shadow = FONT_REGULAR.render(self.text, True, (0, 0, 0))
            text_shadow.set_alpha(shadow_alpha)
            shadow_rect = text_shadow.get_rect(center=(margin + w // 2 + offset[0], margin + h // 2 + offset[1]))
            btn_surface.blit(text_shadow, shadow_rect)
        
        text_surface = FONT_REGULAR.render(self.text, True, text_color)
        text_rect = text_surface.get_rect(center=(margin + w // 2, margin + h // 2))
        btn_surface.blit(text_surface, text_rect)
        
        surface.blit(btn_surface, (x - margin, y - margin))
    
    def is_clicked(self, mouse_pos: Tuple[int, int], mouse_clicked: bool) -> bool:
        """Check if button was clicked."""
        return self.rect.collidepoint(mouse_pos) and mouse_clicked


# =============================================================================
# PARTICLE SYSTEM
# =============================================================================

class Particle:
    """Enhanced particle system with multiple effect types."""
    
    def __init__(self, x: int, y: int, color: Tuple[int, int, int], effect_type: str = 'burst'):
        self.x = x
        self.y = y
        self.color = color
        self.effect_type = effect_type
        
        if effect_type == 'burst':
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 8)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed - 3
            self.size = random.randint(5, 12)
        elif effect_type == 'sparkle':
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            self.size = random.randint(3, 8)
        elif effect_type == 'confetti':
            self.vx = random.uniform(-3, 3)
            self.vy = random.uniform(-6, -2)
            self.size = random.randint(6, 14)
        else:
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed - 2
            self.size = random.randint(4, 10)
        
        self.life = 1.0
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-8, 8)
        self.shape = random.choice(['circle', 'diamond', 'star']) if effect_type != 'confetti' else 'rect'
        self.shimmer = random.uniform(0.8, 1.2)
        
    def update(self):
        """Update particle position and life with physics."""
        self.x += self.vx
        self.y += self.vy
        
        if self.effect_type == 'confetti':
            self.vy += 0.12  # Lighter gravity for confetti
            self.vx *= 0.99
            self.rotation += self.rotation_speed * 2
        else:
            self.vy += 0.18  # Gravity
            self.vx *= 0.97  # Air resistance
            self.rotation += self.rotation_speed
        
        self.life -= 0.012 if self.effect_type == 'confetti' else 0.018
        self.shimmer = 0.7 + 0.3 * math.sin(self.life * 10)
        
    def render(self, surface: pygame.Surface):
        """Render the particle with enhanced glow effect."""
        if self.life > 0:
            # Smooth cubic fade for more elegant disappearance
            alpha = int(255 * self.life * self.life * self.life * self.shimmer)
            alpha = max(0, min(255, alpha))
            size = max(2, int(self.size * (0.5 + 0.5 * self.life)))
            
            # Create particle surface with extra space for glow
            surf_size = size * 4
            particle_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            center = surf_size // 2
            
            if self.shape == 'circle':
                # Multi-layer glow for premium effect
                for i, mult in enumerate([2.5, 2.0, 1.5, 1.0]):
                    glow_alpha = alpha // (4 - i)
                    glow_size = int(size * mult)
                    pygame.draw.circle(particle_surface, (*self.color, glow_alpha), 
                                     (center, center), glow_size)
            elif self.shape == 'diamond':
                # Rotate a square to make diamond
                points = [
                    (center, center - size),
                    (center + size, center),
                    (center, center + size),
                    (center - size, center)
                ]
                pygame.draw.polygon(particle_surface, (*self.color, alpha), points)
            elif self.shape == 'rect':
                # Confetti rectangle
                rect_surf = pygame.Surface((size, size // 2), pygame.SRCALPHA)
                rect_surf.fill((*self.color, alpha))
                rotated = pygame.transform.rotate(rect_surf, self.rotation)
                particle_surface.blit(rotated, rotated.get_rect(center=(center, center)))
            else:  # star
                pygame.draw.circle(particle_surface, (*self.color, alpha), 
                                 (center, center), size)
                # Add cross sparkle
                line_len = size + 2
                pygame.draw.line(particle_surface, (*self.color, alpha // 2), 
                               (center - line_len, center), (center + line_len, center), 2)
                pygame.draw.line(particle_surface, (*self.color, alpha // 2), 
                               (center, center - line_len), (center, center + line_len), 2)
            
            surface.blit(particle_surface, (int(self.x - center), int(self.y - center)))


# =============================================================================
# MAIN GAME CLASS
# =============================================================================

class UnoGame:
    """Main UNO game with professional graphical interface."""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("UNO - Card Game")
        self.clock = pygame.time.Clock()
        
        self.state = GameState.MENU
        self.deck: Optional[Deck] = None
        self.player_hand: List[Card] = []
        self.ai_hand: List[Card] = []
        self.open_card: Optional[Card] = None
        self.particles: List[Particle] = []
        self.message = ""
        self.message_timer = 0
        self.winner = ""
        self.ai_thinking_timer = 0
        self.pending_wild_card: Optional[Card] = None
        self.animation_offset = 0
        
        # RL Agent mode
        self.rl_mode = False
        self.rl_agent = None
        self.rl_thinking_timer = 0
        
        # Model selector for choosing AI opponent
        self.model_selector = ModelSelector(
            WINDOW_WIDTH // 2 - 160, 320, 320, 48, "AI Opponent"
        )
        self.load_rl_agent()
        
        # Create premium styled buttons
        btn_width = 240
        btn_height = 54
        center_x = WINDOW_WIDTH // 2 - btn_width // 2
        
        self.play_button = Button(center_x, 485, btn_width, btn_height, 
                                  "Play Game", COLORS['BUTTON_PRIMARY'])
        self.watch_button = Button(center_x, 555, btn_width, btn_height, 
                                   "Watch AI Play", COLORS['BUTTON_SUCCESS'])
        self.multiplayer_button = Button(center_x, 625, btn_width, btn_height,
                                         "Multiplayer (3-4)", COLORS['ACCENT_PURPLE'])
        self.draw_button = Button(WINDOW_WIDTH // 2 + 180, WINDOW_HEIGHT - 195, 
                                  140, 52, "Draw Card", COLORS['BUTTON_PRIMARY'])
        self.uno_button = Button(WINDOW_WIDTH // 2 + 340, WINDOW_HEIGHT - 195, 
                                 110, 52, "UNO!", COLORS['BUTTON_DANGER'])
        self.menu_button = Button(center_x, 625, btn_width, btn_height, 
                                  "Main Menu", COLORS['DARK_GRAY'])
        
        # Exit button (top right corner) - glass style
        self.exit_button = Button(WINDOW_WIDTH - 100, 20, 80, 36, "Exit", COLORS['DARK_GRAY'], style='glass')
        
        # Color choice buttons with premium styling
        color_btn_size = 100
        color_spacing = 120
        color_start_x = WINDOW_WIDTH // 2 - (color_spacing * 2 - 20)
        self.color_buttons = [
            Button(color_start_x, WINDOW_HEIGHT // 2 - 10, color_btn_size, 60, "Red", COLORS['RED']),
            Button(color_start_x + color_spacing, WINDOW_HEIGHT // 2 - 10, color_btn_size, 60, "Green", COLORS['GRE']),
            Button(color_start_x + color_spacing * 2, WINDOW_HEIGHT // 2 - 10, color_btn_size, 60, "Blue", COLORS['BLU']),
            Button(color_start_x + color_spacing * 3, WINDOW_HEIGHT // 2 - 10, color_btn_size, 60, "Yellow", COLORS['YEL']),
        ]
    
    def load_selected_model(self):
        """Load the currently selected model from the model selector."""
        selected = self.model_selector.get_selected()
        if not selected:
            return
        
        self.sb3_model = None
        self.rl_agent = None
        self.is_recurrent_model = False
        self.lstm_states = None
        self.model_name = selected.name
        
        if selected.algo == 'random':
            print(f"Using Random AI")
            return
        
        if selected.load():
            self.sb3_model = selected.model
            self.is_recurrent_model = selected.is_recurrent
            print(f"✓ Loaded {selected.name}")
        else:
            print(f"Failed to load {selected.name}, using Random AI")
            self.model_name = "Random AI"
    
    def load_rl_agent(self):
        """Load the RL agent - prefer Best Recurrent PPO (60% win rate), fallback to other models."""
        self.sb3_model = None
        self.rl_agent = None
        self.is_recurrent_model = False
        self.lstm_states = None
        self.model_name = "Random AI"

        
        # Get the base directory (where the script is located)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Priority order for models (best performing first)
        model_priority = [
            # Best Recurrent PPO - 60% win rate (BEST)
            ("best_recurrent_ppo_uno.zip", "recurrentppo", "Best Recurrent PPO (60%)"),
            # Optimal Recurrent PPO - 59% win rate
            ("optimal_recurrent_ppo.zip", "recurrentppo", "Optimal Recurrent PPO (59%)"),
            # SB3 Recurrent PPO - 57% win rate
            ("sb3_recurrentppo_uno.zip", "recurrentppo", "SB3 Recurrent PPO (57%)"),
            # Recurrent PPO
            ("recurrent_ppo_uno.zip", "recurrentppo", "Recurrent PPO"),
            # Standard PPO
            ("sb3_ppo_uno.zip", "ppo", "PPO"),
            ("best_ppo_uno.zip", "ppo", "Best PPO"),
            # DQN fallback
            ("sb3_dqn_uno.zip", "dqn", "DQN"),
            # Generic best model
            ("best_model.zip", "ppo", "Best Model"),
        ]
        
        # Try to load models in priority order
        if RECURRENT_PPO_AVAILABLE or SB3_AVAILABLE:
            for filename, algo, name in model_priority:
                model_path = os.path.join(base_dir, "models", filename)
                if os.path.exists(model_path):
                    try:
                        if algo == "recurrentppo" and RECURRENT_PPO_AVAILABLE:
                            self.sb3_model = RecurrentPPO.load(model_path)
                            self.is_recurrent_model = True
                            self.model_name = name
                            print(f"✓ Loaded {name} (LSTM-based, best performance!)")
                            return
                        elif algo == "ppo" and SB3_AVAILABLE:
                            self.sb3_model = PPO.load(model_path)
                            self.is_recurrent_model = False
                            self.model_name = name
                            print(f"✓ Loaded {name}")
                            return
                        elif algo == "dqn" and SB3_AVAILABLE:
                            self.sb3_model = DQN.load(model_path)
                            self.is_recurrent_model = False
                            self.model_name = name
                            print(f"✓ Loaded {name}")
                            return
                    except Exception as e:
                        print(f"Could not load {name}: {e}")
                        continue
            
            print("No trained models found. Will use fallback agent.")
        
        # Fallback to legacy Q-learning agent
        if RL_AVAILABLE:
            try:
                agent_info = {"epsilon": 0.1, "step_step": 0.2}
                self.rl_agent = QLearningAgent(agent_info)
                
                q_values_path = os.path.join(base_dir, "assets", "q-values.csv")
                if os.path.exists(q_values_path):
                    q_df = pd.read_csv(q_values_path, index_col=0)
                    q_df.index = q_df.index.map(lambda x: eval(x) if isinstance(x, str) else x)
                    self.rl_agent.q = q_df
                    print("Loaded trained Q-values (legacy agent)!")
                else:
                    print("No trained Q-values found, using untrained agent.")
            except Exception as e:
                print(f"Could not load legacy RL agent: {e}")
                self.rl_agent = None
    
    def state_to_observation(self, hand: List['Card'], playable: List['Card']) -> np.ndarray:
        """Convert game state to observation vector for SB3 model."""
        # One-hot encode open card color
        color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        color_vec = [0, 0, 0, 0]
        open_color = self.open_card.color if self.open_card.color in color_map else "RED"
        if open_color in color_map:
            color_vec[color_map[open_color]] = 1
        
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
        
        for card in playable:
            if card.color in norm_cards and card.value in range(0, 10):
                play_norm[card.color + "#"] = 1
        
        # Build observation vector (17 features)
        card_values = []
        for color in ["RED", "GRE", "BLU", "YEL"]:
            card_values.append(norm_cards[color] / 2.0)
        for spec in ["SKI", "REV", "PL2"]:
            card_values.append(spec_cards[spec])
        for wild in ["PL4", "COL"]:
            card_values.append(wild_cards[wild])
        for color in ["RED", "GRE", "BLU", "YEL"]:
            card_values.append(play_norm[color + "#"])
        
        obs = np.array(color_vec + card_values, dtype=np.float32)
        
        # Ensure correct size (17 features)
        if len(obs) < 17:
            obs = np.pad(obs, (0, 17 - len(obs)))
        elif len(obs) > 17:
            obs = obs[:17]
        
        return obs
    
    def launch_multiplayer(self):
        """Launch the multiplayer game GUI."""
        import subprocess
        import sys
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        multiplayer_script = os.path.join(base_dir, "multiplayer_gui.py")
        
        if os.path.exists(multiplayer_script):
            # Run in subprocess to avoid conflicts
            pygame.quit()
            subprocess.run([sys.executable, multiplayer_script])
            # Restart this GUI when multiplayer closes
            pygame.init()
            pygame.font.init()
            self.__init__()
        else:
            self.show_message("Multiplayer not installed!")
    
    def start_game(self, rl_mode: bool = False):
        """Initialize a new game."""
        self.rl_mode = rl_mode
        self.deck = Deck()
        self.player_hand = []
        self.ai_hand = []
        self.particles = []
        
        # Reset LSTM states for recurrent models
        self.lstm_states = None
        self.winner = ""
        
        # Deal cards
        for _ in range(7):
            card = self.deck.draw()
            if card:
                self.player_hand.append(card)
            card = self.deck.draw()
            if card:
                self.ai_hand.append(card)
        
        # Draw first open card (must be a number card)
        self.open_card = self.deck.draw()
        while self.open_card and self.open_card.value not in range(0, 10):
            self.deck.cards.insert(0, self.open_card)
            self.deck.shuffle()
            self.open_card = self.deck.draw()
        
        if self.rl_mode:
            self.state = GameState.RL_PLAYER_TURN
            self.rl_thinking_timer = 60
            self.show_message("RL Agent's turn!")
        else:
            self.state = GameState.PLAYER_TURN
            self.show_message("Your turn!")
    
    def show_message(self, text: str, duration: int = 120):
        """Display a message on screen."""
        self.message = text
        self.message_timer = duration
    
    def spawn_particles(self, x: int, y: int, color: Tuple[int, int, int], count: int = 20, effect_type: str = 'burst'):
        """Spawn celebration particles with different effect types."""
        for _ in range(count):
            self.particles.append(Particle(x, y, color, effect_type))
    
    def get_playable_cards(self, hand: List[Card]) -> List[Card]:
        """Get all playable cards from a hand."""
        if not self.open_card:
            return []
        return [card for card in hand 
                if card.evaluate_card(self.open_card.color, self.open_card.value)]
    
    def play_card(self, card: Card, hand: List[Card], is_player: bool = True):
        """Play a card from the given hand."""
        if card in hand:
            hand.remove(card)
            
            # Handle wild cards
            if card.value in ["COL", "PL4"]:
                if is_player and not self.rl_mode:
                    # Human player chooses color via UI
                    self.pending_wild_card = card
                    self.state = GameState.CHOOSING_COLOR
                    return
                elif is_player and self.rl_mode:
                    # RL agent auto-chooses the best color based on hand
                    colors = [c.color for c in self.player_hand if c.color in ["RED", "GRE", "BLU", "YEL"]]
                    if colors:
                        card.color = max(set(colors), key=colors.count)
                    else:
                        card.color = random.choice(["RED", "GRE", "BLU", "YEL"])
                    agent_name = "LSTM Agent" if self.is_recurrent_model else "PPO Agent"
                    self.show_message(f"{agent_name} chooses {card.color}")
                else:
                    # AI opponent chooses most common color
                    colors = [c.color for c in self.ai_hand if c.color in ["RED", "GRE", "BLU", "YEL"]]
                    if colors:
                        card.color = max(set(colors), key=colors.count)
                    else:
                        card.color = random.choice(["RED", "GRE", "BLU", "YEL"])
            
            # Handle +4 and +2
            if card.value == "PL4":
                if is_player:
                    for _ in range(4):
                        drawn = self.deck.draw()
                        if drawn:
                            self.ai_hand.append(drawn)
                    self.show_message("AI draws 4 cards!")
                else:
                    for _ in range(4):
                        drawn = self.deck.draw()
                        if drawn:
                            self.player_hand.append(drawn)
                    self.show_message("You draw 4 cards!")
            elif card.value == "PL2":
                if is_player:
                    for _ in range(2):
                        drawn = self.deck.draw()
                        if drawn:
                            self.ai_hand.append(drawn)
                    self.show_message("AI draws 2 cards!")
                else:
                    for _ in range(2):
                        drawn = self.deck.draw()
                        if drawn:
                            self.player_hand.append(drawn)
                    self.show_message("You draw 2 cards!")
            
            # Discard old open card and set new one
            if self.open_card:
                self.deck.discard(self.open_card)
            self.open_card = card
            
            # Particles effect
            self.spawn_particles(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, card.get_color_rgb())
            
            # Check for winner
            if len(hand) == 0:
                if is_player:
                    self.winner = "RL Agent" if self.rl_mode else "You"
                else:
                    self.winner = "AI"
                self.state = GameState.GAME_OVER
                return
            
            # Handle skip and reverse (skip opponent's turn)
            if card.value in ["SKI", "REV"]:
                if is_player:
                    self.show_message("AI's turn skipped!")
                    if self.rl_mode:
                        self.state = GameState.RL_PLAYER_TURN
                        self.rl_thinking_timer = 60
                    else:
                        self.state = GameState.PLAYER_TURN
                else:
                    if self.rl_mode:
                        self.show_message("RL Agent's turn skipped!")
                    else:
                        self.show_message("Your turn skipped!")
                    self.ai_thinking_timer = 60
                    self.state = GameState.AI_TURN
            else:
                # Switch turns
                if is_player:
                    self.ai_thinking_timer = 60
                    self.state = GameState.AI_TURN
                    self.show_message("AI is thinking...")
                else:
                    if self.rl_mode:
                        self.state = GameState.RL_PLAYER_TURN
                        self.rl_thinking_timer = 60
                        self.show_message("RL Agent's turn!")
                    else:
                        self.state = GameState.PLAYER_TURN
                        self.show_message("Your turn!")
    
    def ai_turn(self):
        """Execute AI's turn."""
        playable = self.get_playable_cards(self.ai_hand)
        
        if playable:
            # AI strategy: prioritize action cards, then match color, then wild
            def card_priority(card):
                if card.value in ["SKI", "REV", "PL2"]:
                    return 0
                elif card.color == self.open_card.color:
                    return 1
                elif card.value == self.open_card.value:
                    return 2
                elif card.value == "PL4":
                    return 3
                elif card.value == "COL":
                    return 4
                return 5
            
            playable.sort(key=card_priority)
            self.play_card(playable[0], self.ai_hand, is_player=False)
            self.show_message(f"AI plays {playable[0].color} {playable[0].get_display_value()}")
        else:
            # Draw a card
            drawn = self.deck.draw()
            if drawn:
                self.ai_hand.append(drawn)
                self.show_message("AI draws a card")
                
                # Check if drawn card is playable
                if drawn.evaluate_card(self.open_card.color, self.open_card.value):
                    self.ai_thinking_timer = 60
                else:
                    if self.rl_mode:
                        self.state = GameState.RL_PLAYER_TURN
                        self.rl_thinking_timer = 60
                        self.show_message("🤖 PPO Agent's turn!")
                    else:
                        self.state = GameState.PLAYER_TURN
                        self.show_message("Your turn!")
            else:
                if self.rl_mode:
                    self.state = GameState.RL_PLAYER_TURN
                    self.rl_thinking_timer = 60
                    self.show_message("🤖 PPO Agent's turn!")
                else:
                    self.state = GameState.PLAYER_TURN
                    self.show_message("Your turn!")
    
    def get_state_dict(self, hand: List[Card], playable: List[Card]) -> Dict:
        """Build state dictionary for RL agent from current hand."""
        norm_cards = {"RED": 2, "GRE": 2, "BLU": 2, "YEL": 2}
        spec_cards = {"SKI": 1, "REV": 1, "PL2": 1}
        wild_cards = {"PL4": 1, "COL": 1}
        
        state = {}
        state["OPEN"] = self.open_card.color if self.open_card.color in ["RED", "GRE", "BLU", "YEL"] else random.choice(["RED", "GRE", "BLU", "YEL"])
        
        # Normal hand cards
        for key, val in norm_cards.items():
            count = sum(1 for card in hand if card.color == key and card.value in range(0, 10))
            state[key] = min(count, val)
        
        # Special hand cards
        for key, val in spec_cards.items():
            count = sum(1 for card in hand if card.value == key)
            state[key] = min(count, val)
        
        # Wild hand cards
        for key, val in wild_cards.items():
            count = sum(1 for card in hand if card.value == key)
            state[key] = min(count, val)
        
        # Normal playable cards
        for key, val in norm_cards.items():
            count = sum(1 for card in playable if card.color == key and card.value in range(0, 10))
            state[key + "#"] = min(count, val - 1)
        
        # Special playable cards
        for key, val in spec_cards.items():
            count = sum(1 for card in playable if card.value == key)
            state[key + "#"] = min(count, val)
        
        return state
    
    def get_actions_dict(self, playable: List[Card]) -> Dict:
        """Build actions dictionary for RL agent from playable cards."""
        norm_cards = {"RED": 1, "GRE": 1, "BLU": 1, "YEL": 1}
        spec_cards = {"SKI": 1, "REV": 1, "PL2": 1}
        wild_cards = {"PL4": 1, "COL": 1}
        
        actions = {}
        
        # Normal playable cards
        for key in norm_cards.keys():
            actions[key] = min(sum(1 for card in playable if card.color == key and card.value in range(0, 10)), 1)
        
        # Special playable cards
        for key in spec_cards.keys():
            actions[key] = min(sum(1 for card in playable if card.value == key), 1)
        
        # Wild playable cards
        for key in wild_cards.keys():
            actions[key] = min(sum(1 for card in playable if card.value == key), 1)
        
        return actions
    
    def find_card_for_action(self, action: str, hand: List[Card], playable: List[Card]) -> Optional[Card]:
        """Find a card in hand that matches the RL agent's chosen action."""
        # Wild cards
        if action in ["COL", "PL4"]:
            for card in hand:
                if card.value == action:
                    return card
        
        # Normal cards with different color (match by value)
        elif action in ["RED", "GRE", "BLU", "YEL"] and action != self.open_card.color:
            for card in hand:
                if card.color == action and card.value == self.open_card.value:
                    return card
            # Fallback: any card of that color
            for card in playable:
                if card.color == action and card.value in range(0, 10):
                    return card
        
        # Normal cards with same color
        elif action in ["RED", "GRE", "BLU", "YEL"] and action == self.open_card.color:
            for card in hand:
                if card.color == action and card.value in range(0, 10):
                    return card
        
        # Special cards with same color
        elif action not in ["RED", "GRE", "BLU", "YEL"] and action != self.open_card.value:
            for card in hand:
                if card.color == self.open_card.color and card.value == action:
                    return card
        
        # Special cards with different color (match by value)
        else:
            for card in hand:
                if card.value == action:
                    return card
        
        return None
    
    def sb3_choose_action(self, hand: List['Card'], playable: List['Card']) -> Optional['Card']:
        """Use SB3 model to choose an action and return the card to play."""
        action_names = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        
        # Get observation for the model
        obs = self.state_to_observation(hand, playable)
        
        # Get action from model (handle recurrent vs non-recurrent)
        if self.is_recurrent_model:
            # Recurrent model needs LSTM states
            episode_start = np.array([False]) if self.lstm_states is not None else np.array([True])
            action, self.lstm_states = self.sb3_model.predict(
                obs, 
                state=self.lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
        else:
            action, _ = self.sb3_model.predict(obs, deterministic=True)
        action = int(action)
        
        # Map action to action name
        action_name = action_names[action]
        
        # Find corresponding card
        card = self.find_card_for_action(action_name, hand, playable)
        
        # If no matching card, try other valid actions
        if card is None:
            for card_option in playable:
                return card_option
        
        return card
    
    def rl_player_turn(self):
        """Execute RL agent's turn as the player (using SB3 or legacy agent)."""
        # Check if we have an SB3 model (preferred)
        if self.sb3_model is not None:
            self._sb3_player_turn()
            return
        
        # Fallback to legacy Q-learning agent
        if self.rl_agent is None:
            self.ai_turn_for_player()
            return
        
        playable = self.get_playable_cards(self.player_hand)
        
        if playable:
            # Get state and actions for RL agent
            state_dict = self.get_state_dict(self.player_hand, playable)
            actions_dict = self.get_actions_dict(playable)
            
            # Let RL agent choose action
            try:
                action = self.rl_agent.step(state_dict, actions_dict)
                card = self.find_card_for_action(action, self.player_hand, playable)
                
                if card:
                    self.show_message(f"RL Agent plays {card.color} {card.get_display_value()}")
                    self.play_card(card, self.player_hand, is_player=True)
                else:
                    # Fallback: play first playable card
                    card = playable[0]
                    self.show_message(f"RL Agent plays {card.color} {card.get_display_value()}")
                    self.play_card(card, self.player_hand, is_player=True)
            except Exception as e:
                print(f"RL agent error: {e}")
                # Fallback
                card = playable[0]
                self.show_message(f"RL Agent plays {card.color} {card.get_display_value()}")
                self.play_card(card, self.player_hand, is_player=True)
        else:
            # Draw a card
            drawn = self.deck.draw()
            if drawn:
                self.player_hand.append(drawn)
                self.show_message(f"RL Agent draws a card")
                
                # Check if drawn card is playable
                if drawn.evaluate_card(self.open_card.color, self.open_card.value):
                    self.rl_thinking_timer = 60
                    self.state = GameState.RL_PLAYER_TURN
                else:
                    self.ai_thinking_timer = 60
                    self.state = GameState.AI_TURN
                    self.show_message("AI is thinking...")
            else:
                self.ai_thinking_timer = 60
                self.state = GameState.AI_TURN
                self.show_message("AI is thinking...")
    
    def _sb3_player_turn(self):
        """Execute turn using SB3 trained model."""
        playable = self.get_playable_cards(self.player_hand)
        
        if playable:
            try:
                card = self.sb3_choose_action(self.player_hand, playable)
                
                if card:
                    agent_name = "LSTM Agent" if self.is_recurrent_model else "PPO Agent"
                    self.show_message(f"{agent_name} plays {card.color} {card.get_display_value()}")
                    self.play_card(card, self.player_hand, is_player=True)
                else:
                    # Fallback
                    card = playable[0]
                    self.show_message(f"Agent plays {card.color} {card.get_display_value()}")
                    self.play_card(card, self.player_hand, is_player=True)
            except Exception as e:
                print(f"SB3 agent error: {e}")
                card = playable[0]
                self.show_message(f"Agent plays {card.color} {card.get_display_value()}")
                self.play_card(card, self.player_hand, is_player=True)
        else:
            # Draw a card
            drawn = self.deck.draw()
            if drawn:
                self.player_hand.append(drawn)
                agent_name = "LSTM Agent" if self.is_recurrent_model else "PPO Agent"
                self.show_message(f"{agent_name} draws a card")
                
                if drawn.evaluate_card(self.open_card.color, self.open_card.value):
                    # Play the drawn card immediately
                    self.rl_thinking_timer = 60
                    self.state = GameState.RL_PLAYER_TURN
                else:
                    # Pass turn to AI
                    self.ai_thinking_timer = 60
                    self.state = GameState.AI_TURN
                    self.show_message("AI is thinking...")
            else:
                self.ai_thinking_timer = 60
                self.state = GameState.AI_TURN
                self.show_message("AI is thinking...")
    
    def handle_events(self):
        """Handle all pygame events."""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state != GameState.MENU:
                        self.state = GameState.MENU
            
            # Handle model selector events in menu
            if self.state == GameState.MENU:
                if self.model_selector.handle_event(event):
                    # Model changed - reload
                    self.load_selected_model()
        
        # Update button hover states
        self.play_button.update(mouse_pos)
        self.watch_button.update(mouse_pos)
        self.multiplayer_button.update(mouse_pos)
        self.draw_button.update(mouse_pos)
        self.uno_button.update(mouse_pos)
        self.menu_button.update(mouse_pos)
        self.exit_button.update(mouse_pos)
        for btn in self.color_buttons:
            btn.update(mouse_pos)
        
        # Check exit button click (works in all states)
        if self.exit_button.is_clicked(mouse_pos, mouse_clicked):
            return False
        
        # Handle menu state
        if self.state == GameState.MENU:
            if self.play_button.is_clicked(mouse_pos, mouse_clicked):
                self.load_selected_model()  # Ensure latest selection is loaded
                self.start_game(rl_mode=False)
            elif self.watch_button.is_clicked(mouse_pos, mouse_clicked):
                self.load_selected_model()  # Ensure latest selection is loaded
                if self.sb3_model is not None or (RL_AVAILABLE and self.rl_agent):
                    self.start_game(rl_mode=True)
                else:
                    self.show_message("Select a trained model!")
            elif self.multiplayer_button.is_clicked(mouse_pos, mouse_clicked):
                # Launch multiplayer GUI
                self.launch_multiplayer()
        
        # Handle game over state
        elif self.state == GameState.GAME_OVER:
            if self.menu_button.is_clicked(mouse_pos, mouse_clicked):
                self.state = GameState.MENU
            elif self.play_button.is_clicked(mouse_pos, mouse_clicked):
                self.start_game(rl_mode=self.rl_mode)
        
        # Handle color choosing
        elif self.state == GameState.CHOOSING_COLOR:
            colors = ["RED", "GRE", "BLU", "YEL"]
            for i, btn in enumerate(self.color_buttons):
                if btn.is_clicked(mouse_pos, mouse_clicked):
                    if self.pending_wild_card:
                        self.pending_wild_card.color = colors[i]
                        
                        # Handle +4
                        if self.pending_wild_card.value == "PL4":
                            for _ in range(4):
                                drawn = self.deck.draw()
                                if drawn:
                                    self.ai_hand.append(drawn)
                            self.show_message("AI draws 4 cards!")
                        
                        if self.open_card:
                            self.deck.discard(self.open_card)
                        self.open_card = self.pending_wild_card
                        self.pending_wild_card = None
                        
                        # Check for winner
                        if len(self.player_hand) == 0:
                            self.winner = "RL Agent" if self.rl_mode else "You"
                            self.state = GameState.GAME_OVER
                        else:
                            self.ai_thinking_timer = 60
                            self.state = GameState.AI_TURN
                            self.show_message("AI is thinking...")
        
        # Handle player turn
        elif self.state == GameState.PLAYER_TURN:
            # Check card clicks - use same dimensions as render_game
            card_width = min(80, (WINDOW_WIDTH - 320) // max(len(self.player_hand), 1))
            start_x = (WINDOW_WIDTH - card_width * len(self.player_hand)) // 2
            y = WINDOW_HEIGHT - 170
            
            playable = self.get_playable_cards(self.player_hand)
            
            scaled_card_w = int(CARD_WIDTH * 0.92)
            scaled_card_h = int(CARD_HEIGHT * 0.92)
            
            for i, card in enumerate(self.player_hand):
                hover_offset = 28 if card.hover else 0
                playable_offset = 10 if card in playable else 0
                card_y = y - hover_offset - playable_offset
                card_x = start_x + i * card_width
                card_rect = pygame.Rect(card_x, card_y, scaled_card_w, scaled_card_h)
                
                # Hover effect
                card.hover = card_rect.collidepoint(mouse_pos) and card in playable
                
                if mouse_clicked and card_rect.collidepoint(mouse_pos):
                    if card in playable:
                        self.play_card(card, self.player_hand)
            
            # Draw button
            if self.draw_button.is_clicked(mouse_pos, mouse_clicked):
                drawn = self.deck.draw()
                if drawn:
                    self.player_hand.append(drawn)
                    self.show_message(f"You drew {drawn.color} {drawn.get_display_value()}")
                    
                    # Check if drawn card is playable
                    if not drawn.evaluate_card(self.open_card.color, self.open_card.value):
                        self.ai_thinking_timer = 60
                        self.state = GameState.AI_TURN
                        self.show_message("AI is thinking...")
        
        return True
    
    def update(self):
        """Update game state."""
        self.animation_offset = (self.animation_offset + 1) % 360
        
        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= 1
        
        # Update particles
        for particle in self.particles[:]:
            particle.update()
            if particle.life <= 0:
                self.particles.remove(particle)
        
        # Handle AI turn
        if self.state == GameState.AI_TURN:
            if self.ai_thinking_timer > 0:
                self.ai_thinking_timer -= 1
            else:
                self.ai_turn()
        
        # Handle RL player turn
        if self.state == GameState.RL_PLAYER_TURN:
            if self.rl_thinking_timer > 0:
                self.rl_thinking_timer -= 1
            else:
                self.rl_player_turn()
    
    def render_background(self):
        """Render ultra-modern gradient background with ambient lighting effects."""
        # Create smooth gradient with multiple stops
        bg_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Multi-stop vertical gradient for depth
        for y in range(WINDOW_HEIGHT):
            progress = y / WINDOW_HEIGHT
            # Smooth cubic easing for elegant gradient
            eased = progress * progress * (3 - 2 * progress)
            
            # Add subtle color variation
            r = int(COLORS['BG_TOP'][0] + eased * (COLORS['BG_BOTTOM'][0] - COLORS['BG_TOP'][0]))
            g = int(COLORS['BG_TOP'][1] + eased * (COLORS['BG_BOTTOM'][1] - COLORS['BG_TOP'][1]))
            b = int(COLORS['BG_TOP'][2] + eased * (COLORS['BG_BOTTOM'][2] - COLORS['BG_TOP'][2]))
            pygame.draw.line(bg_surface, (r, g, b), (0, y), (WINDOW_WIDTH, y))
        
        self.screen.blit(bg_surface, (0, 0))
        
        # Modern animated mesh gradient orbs
        orb_configs = [
            (COLORS['ACCENT_PURPLE'], 0.15, 280, 0.5),
            (COLORS['ACCENT_CYAN'], 0.12, 240, 0.6),
            (COLORS['ACCENT_BLUE'], 0.18, 220, 0.4),
            (COLORS['ACCENT_PINK'], 0.10, 200, 0.7),
        ]
        
        for i, (color, speed, radius, vert_scale) in enumerate(orb_configs):
            angle = math.radians(self.animation_offset * speed + i * 90)
            cx = WINDOW_WIDTH / 2 + math.cos(angle) * (280 + i * 70)
            cy = WINDOW_HEIGHT / 2 + math.sin(angle) * (150 + i * 40) * vert_scale
            
            # Multi-layer soft gradient orb
            glow_surface = pygame.Surface((radius * 3, radius * 3), pygame.SRCALPHA)
            for rad in range(radius, 10, -6):
                alpha = int(18 * (radius - rad) / radius)
                pygame.draw.circle(glow_surface, (*color[:3], alpha), 
                                 (radius * 1.5, radius * 1.5), rad)
            
            self.screen.blit(glow_surface, (int(cx - radius * 1.5), int(cy - radius * 1.5)))
        
        # Subtle radial vignette from center
        vignette = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        max_rad = int(max(WINDOW_WIDTH, WINDOW_HEIGHT) * 0.8)
        for rad in range(max_rad, 200, -15):
            alpha = int(4 * (max_rad - rad) / (max_rad - 200))
            pygame.draw.circle(vignette, (0, 0, 0, alpha), 
                             (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2), rad)
        self.screen.blit(vignette, (0, 0))
        
        # Modern grid pattern overlay (very subtle)
        grid_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        grid_spacing = 80
        grid_alpha = 8
        for gx in range(0, WINDOW_WIDTH, grid_spacing):
            pygame.draw.line(grid_surface, (255, 255, 255, grid_alpha), (gx, 0), (gx, WINDOW_HEIGHT))
        for gy in range(0, WINDOW_HEIGHT, grid_spacing):
            pygame.draw.line(grid_surface, (255, 255, 255, grid_alpha), (0, gy), (WINDOW_WIDTH, gy))
        self.screen.blit(grid_surface, (0, 0))
    
    def render_menu(self):
        """Render the main menu with premium modern design."""
        self.render_background()
        
        # Floating ambient particles in menu
        if random.random() < 0.03:
            px = random.randint(100, WINDOW_WIDTH - 100)
            py = WINDOW_HEIGHT + 10
            color = random.choice([COLORS['ACCENT_PURPLE'], COLORS['ACCENT_BLUE'], COLORS['ACCENT_CYAN']])
            self.particles.append(Particle(px, py, color, 'sparkle'))
        
        # Main title with multi-layer glow effect
        title_text = "UNO"
        title_y = 130
        
        # Outer glow layers
        glow_colors = [
            (*COLORS['ACCENT_PURPLE'][:3], 40),
            (*COLORS['ACCENT_BLUE'][:3], 60),
        ]
        
        for i, glow_col in enumerate(glow_colors):
            glow_size = 8 - i * 3
            for ox, oy in [(-glow_size, 0), (glow_size, 0), (0, -glow_size), (0, glow_size),
                          (-glow_size//2, -glow_size//2), (glow_size//2, glow_size//2)]:
                glow_text = FONT_TITLE.render(title_text, True, glow_col[:3])
                glow_rect = glow_text.get_rect(center=(WINDOW_WIDTH // 2 + ox, title_y + oy))
                glow_surf = pygame.Surface(glow_text.get_size(), pygame.SRCALPHA)
                glow_surf.blit(glow_text, (0, 0))
                glow_surf.set_alpha(glow_col[3])
                self.screen.blit(glow_surf, glow_rect)
        
        # Main title
        title_main = FONT_TITLE.render(title_text, True, COLORS['WHITE'])
        title_rect = title_main.get_rect(center=(WINDOW_WIDTH // 2, title_y))
        self.screen.blit(title_main, title_rect)
        
        # Subtitle with fade effect
        subtitle = FONT_SUBTITLE.render("REINFORCEMENT LEARNING EDITION", True, COLORS['MUTED'])
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, title_y + 55))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Animated floating cards display with enhanced effects
        card_colors = [COLORS['RED'], COLORS['GRE'], COLORS['BLU'], COLORS['YEL']]
        card_values = ['7', '⊘', '⟲', '+4']
        card_spacing = 100
        cards_start_x = WINDOW_WIDTH // 2 - (len(card_colors) - 1) * card_spacing // 2
        
        for i, (color, val) in enumerate(zip(card_colors, card_values)):
            # Smooth floating animation with phase offset
            float_offset = math.sin(math.radians(self.animation_offset * 1.2 + i * 70)) * 15
            rotation = math.sin(math.radians(self.animation_offset * 0.8 + i * 90)) * 8
            scale_pulse = 1.0 + 0.03 * math.sin(math.radians(self.animation_offset * 2 + i * 45))
            
            x = cards_start_x + i * card_spacing - 35
            y = 260 + int(float_offset)
            
            # Create card surface with premium styling
            base_w, base_h = 75, 110
            card_w = int(base_w * scale_pulse)
            card_h = int(base_h * scale_pulse)
            card_surface = pygame.Surface((card_w + 20, card_h + 20), pygame.SRCALPHA)
            
            # Multi-layer shadow for depth
            for s in range(3):
                shadow_alpha = 25 - s * 6
                shadow_offset = 6 + s * 2
                pygame.draw.rect(card_surface, (0, 0, 0, shadow_alpha), 
                               (10 + s, shadow_offset + s, card_w, card_h), border_radius=12)
            
            # Card body with gradient effect
            pygame.draw.rect(card_surface, color, (10, 10, card_w, card_h), border_radius=12)
            
            # Top highlight
            highlight_surf = pygame.Surface((card_w - 8, card_h // 3), pygame.SRCALPHA)
            for hy in range(card_h // 3):
                alpha = int(40 * (1 - hy / (card_h // 3)))
                pygame.draw.line(highlight_surf, (255, 255, 255, alpha), (0, hy), (card_w - 8, hy))
            card_surface.blit(highlight_surf, (14, 12))
            
            # Inner white oval with rotation
            oval_surface = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.ellipse(oval_surface, (255, 255, 255, 230), 
                              (10, 20, card_w - 20, card_h - 40))
            rotated_oval = pygame.transform.rotate(oval_surface, 28)
            oval_pos = (10 + card_w // 2 - rotated_oval.get_width() // 2,
                       10 + card_h // 2 - rotated_oval.get_height() // 2)
            card_surface.blit(rotated_oval, oval_pos)
            
            # Value with shadow
            text_shadow = FONT_CARD.render(val, True, (0, 0, 0))
            text = FONT_CARD.render(val, True, color)
            text_rect = text.get_rect(center=(10 + card_w // 2, 10 + card_h // 2))
            card_surface.blit(text_shadow, (text_rect.x + 1, text_rect.y + 1))
            card_surface.blit(text, text_rect)
            
            # Apply rotation and blit
            rotated_card = pygame.transform.rotate(card_surface, rotation)
            card_pos = (x + card_w // 2 - rotated_card.get_width() // 2,
                       y - rotated_card.get_height() // 2 + card_h // 2)
            self.screen.blit(rotated_card, card_pos)
        
        # Glass panel for buttons
        panel_w, panel_h = 280, 240
        panel_x = WINDOW_WIDTH // 2 - panel_w // 2
        panel_y = 420
        
        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (255, 255, 255, 8), (0, 0, panel_w, panel_h), border_radius=20)
        pygame.draw.rect(panel_surface, (255, 255, 255, 25), (0, 0, panel_w, panel_h), width=1, border_radius=20)
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Update button positions for new layout
        btn_width = 240
        btn_height = 54
        center_x = WINDOW_WIDTH // 2 - btn_width // 2
        
        self.play_button.rect = pygame.Rect(center_x, 485, btn_width, btn_height)
        self.watch_button.rect = pygame.Rect(center_x, 555, btn_width, btn_height)
        self.multiplayer_button.rect = pygame.Rect(center_x, 625, btn_width, btn_height)
        
        # Draw model selector first (above buttons)
        self.model_selector.draw(self.screen)
        
        # Play button
        self.play_button.render(self.screen)
        
        # Watch AI button
        self.watch_button.render(self.screen)
        
        # Multiplayer button
        self.multiplayer_button.render(self.screen)
        
        # Exit button with updated position
        self.exit_button.rect = pygame.Rect(WINDOW_WIDTH - 100, 20, 80, 36)
        self.exit_button.render(self.screen)
        
        # Instructions panel at bottom with glass effect
        inst_panel_y = 700
        inst_panel_w = 500
        inst_panel_h = 45
        inst_surface = pygame.Surface((inst_panel_w, inst_panel_h), pygame.SRCALPHA)
        pygame.draw.rect(inst_surface, (0, 0, 0, 50), (0, 0, inst_panel_w, inst_panel_h), border_radius=12)
        pygame.draw.rect(inst_surface, (255, 255, 255, 15), (0, 0, inst_panel_w, inst_panel_h), width=1, border_radius=12)
        self.screen.blit(inst_surface, (WINDOW_WIDTH // 2 - inst_panel_w // 2, inst_panel_y))
        
        instructions = "Match cards by color or number • Special cards: Skip, Reverse, +2, Wild, +4"
        inst = FONT_TINY.render(instructions, True, COLORS['LIGHT_GRAY'])
        inst_rect = inst.get_rect(center=(WINDOW_WIDTH // 2, inst_panel_y + 22))
        self.screen.blit(inst, inst_rect)
        
        # Render particles
        for particle in self.particles:
            particle.render(self.screen)
    
    def render_game(self):
        """Render the game state with premium modern UI."""
        self.render_background()
        
        # === AI SECTION (Top) ===
        # Glass panel background for AI with gradient border
        ai_panel = pygame.Surface((WINDOW_WIDTH - 60, 125), pygame.SRCALPHA)
        pygame.draw.rect(ai_panel, (0, 0, 0, 40), (0, 0, WINDOW_WIDTH - 60, 125), border_radius=18)
        pygame.draw.rect(ai_panel, (255, 255, 255, 15), (0, 0, WINDOW_WIDTH - 60, 125), width=1, border_radius=18)
        self.screen.blit(ai_panel, (30, 15))
        
        # Render AI hand (face down) with stacking effect
        ai_card_width = min(60, (WINDOW_WIDTH - 350) // max(len(self.ai_hand), 1))
        ai_start_x = (WINDOW_WIDTH - ai_card_width * len(self.ai_hand)) // 2
        
        for i, card in enumerate(self.ai_hand):
            card.render(self.screen, ai_start_x + i * ai_card_width, 40, face_up=False, scale=0.68)
        
        # AI label with premium badge
        ai_badge = pygame.Surface((160, 34), pygame.SRCALPHA)
        pygame.draw.rect(ai_badge, (*COLORS['ACCENT_ROSE'][:3], 200), (0, 0, 160, 34), border_radius=17)
        pygame.draw.rect(ai_badge, (255, 255, 255, 40), (0, 0, 160, 34), width=1, border_radius=17)
        ai_text = FONT_SMALL.render(f"AI - {len(self.ai_hand)} cards", True, COLORS['WHITE'])
        ai_text_rect = ai_text.get_rect(center=(80, 17))
        ai_badge.blit(ai_text, ai_text_rect)
        self.screen.blit(ai_badge, (40, 25))
        
        # === CENTER PLAY AREA ===
        # Premium center panel with glass effect
        center_panel = pygame.Surface((440, 200), pygame.SRCALPHA)
        pygame.draw.rect(center_panel, (0, 0, 0, 35), (0, 0, 440, 200), border_radius=24)
        pygame.draw.rect(center_panel, (255, 255, 255, 12), (0, 0, 440, 200), width=1, border_radius=24)
        self.screen.blit(center_panel, (WINDOW_WIDTH // 2 - 220, WINDOW_HEIGHT // 2 - 100))
        
        # Render deck with premium styling
        deck_x = WINDOW_WIDTH // 2 - 135
        deck_y = WINDOW_HEIGHT // 2 - 68
        
        # Draw deck stack effect with subtle glow
        deck_count = len(self.deck.cards) if self.deck else 0
        
        # Deck glow
        deck_glow = pygame.Surface((CARD_WIDTH + 40, CARD_HEIGHT + 40), pygame.SRCALPHA)
        pygame.draw.rect(deck_glow, (*COLORS['ACCENT_INDIGO'][:3], 20), 
                        (0, 0, CARD_WIDTH + 40, CARD_HEIGHT + 40), border_radius=18)
        self.screen.blit(deck_glow, (deck_x - 20, deck_y - 20))
        
        for i in range(min(5, deck_count)):
            offset = i * 2
            temp_card = Card("WILD", "?")
            temp_card.render(self.screen, deck_x - offset, deck_y - offset, face_up=False, scale=1.0)
        
        # Deck count badge with glass effect
        if deck_count > 0:
            count_badge = pygame.Surface((55, 30), pygame.SRCALPHA)
            pygame.draw.rect(count_badge, (*COLORS['ACCENT_BLUE'][:3], 200), (0, 0, 55, 30), border_radius=15)
            pygame.draw.rect(count_badge, (255, 255, 255, 50), (0, 0, 55, 30), width=1, border_radius=15)
            count_text = FONT_SMALL.render(str(deck_count), True, COLORS['WHITE'])
            count_text_rect = count_text.get_rect(center=(55 // 2, 15))
            count_badge.blit(count_text, count_text_rect)
            self.screen.blit(count_badge, (deck_x + 18, deck_y + CARD_HEIGHT + 10))
        
        # Render open card with animated glow
        if self.open_card:
            open_x = WINDOW_WIDTH // 2 + 40
            open_y = WINDOW_HEIGHT // 2 - 68
            
            # Pulsing glow under open card
            glow_intensity = 0.7 + 0.3 * math.sin(math.radians(self.animation_offset * 2))
            glow_color = self.open_card.get_color_rgb()
            
            glow_surface = pygame.Surface((CARD_WIDTH + 50, CARD_HEIGHT + 50), pygame.SRCALPHA)
            for i in range(3):
                alpha = int(30 * glow_intensity / (i + 1))
                glow_rect = pygame.Rect(i * 5, i * 5, CARD_WIDTH + 50 - i * 10, CARD_HEIGHT + 50 - i * 10)
                pygame.draw.rect(glow_surface, (*glow_color, alpha), glow_rect, border_radius=20 - i * 2)
            self.screen.blit(glow_surface, (open_x - 25, open_y - 25))
            
            self.open_card.render(self.screen, open_x, open_y, scale=1.0)
        
        # === PLAYER SECTION (Bottom) ===
        # Premium glass panel for player
        player_panel = pygame.Surface((WINDOW_WIDTH - 60, 175), pygame.SRCALPHA)
        pygame.draw.rect(player_panel, (0, 0, 0, 40), (0, 0, WINDOW_WIDTH - 60, 175), border_radius=18)
        pygame.draw.rect(player_panel, (255, 255, 255, 15), (0, 0, WINDOW_WIDTH - 60, 175), width=1, border_radius=18)
        self.screen.blit(player_panel, (30, WINDOW_HEIGHT - 190))
        
        # Render player hand with hover effects
        playable = self.get_playable_cards(self.player_hand)
        card_width = min(80, (WINDOW_WIDTH - 320) // max(len(self.player_hand), 1))
        start_x = (WINDOW_WIDTH - card_width * len(self.player_hand)) // 2
        y = WINDOW_HEIGHT - 170
        
        for i, card in enumerate(self.player_hand):
            hover_offset = 28 if card.hover else 0
            playable_offset = 10 if card in playable else 0
            card_y = y - hover_offset - playable_offset
            card.render(self.screen, start_x + i * card_width, card_y, scale=0.92)
            
            # Subtle glow for playable cards
            if card in playable and self.state == GameState.PLAYER_TURN:
                glow_surf = pygame.Surface((int(CARD_WIDTH * 0.92) + 12, int(CARD_HEIGHT * 0.92) + 12), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (255, 255, 255, 20), 
                               (0, 0, int(CARD_WIDTH * 0.92) + 12, int(CARD_HEIGHT * 0.92) + 12), 
                               border_radius=14)
                self.screen.blit(glow_surf, (start_x + i * card_width - 6, card_y - 6))
            
            # Dim non-playable cards with vignette effect
            if card not in playable and self.state == GameState.PLAYER_TURN:
                dim_surface = pygame.Surface((int(CARD_WIDTH * 0.92) + 10, int(CARD_HEIGHT * 0.92) + 10), pygame.SRCALPHA)
                dim_surface.fill((0, 0, 0, 140))
                self.screen.blit(dim_surface, (start_x + i * card_width, card_y))
        
        # Player label badge with glass effect
        if self.rl_mode:
            badge_color = COLORS['ACCENT_GREEN']
            label_text = f"PPO Agent - {len(self.player_hand)} cards"
        else:
            badge_color = COLORS['ACCENT_BLUE']
            label_text = f"You - {len(self.player_hand)} cards"
        
        player_badge = pygame.Surface((185, 34), pygame.SRCALPHA)
        pygame.draw.rect(player_badge, (*badge_color[:3], 200), (0, 0, 185, 34), border_radius=17)
        pygame.draw.rect(player_badge, (255, 255, 255, 40), (0, 0, 185, 34), width=1, border_radius=17)
        player_text = FONT_SMALL.render(label_text, True, COLORS['WHITE'])
        player_text_rect = player_text.get_rect(center=(185 // 2, 17))
        player_badge.blit(player_text, player_text_rect)
        self.screen.blit(player_badge, (40, WINDOW_HEIGHT - 52))
        
        # Draw and UNO buttons with updated positions
        if self.state == GameState.PLAYER_TURN and not self.rl_mode:
            self.draw_button.rect = pygame.Rect(WINDOW_WIDTH // 2 + 200, WINDOW_HEIGHT - 195, 140, 52)
            self.draw_button.render(self.screen)
            if len(self.player_hand) == 2:
                self.uno_button.rect = pygame.Rect(WINDOW_WIDTH // 2 + 360, WINDOW_HEIGHT - 195, 110, 52)
                self.uno_button.render(self.screen)
        
        # Color choosing overlay with premium modal
        if self.state == GameState.CHOOSING_COLOR:
            # Animated backdrop blur simulation
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 220))
            self.screen.blit(overlay, (0, 0))
            
            # Premium modal panel with glow
            modal_w, modal_h = 560, 180
            modal_x = WINDOW_WIDTH // 2 - modal_w // 2
            modal_y = WINDOW_HEIGHT // 2 - modal_h // 2 - 20
            
            # Outer glow
            glow_surf = pygame.Surface((modal_w + 40, modal_h + 40), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*COLORS['ACCENT_PURPLE'][:3], 30), 
                           (0, 0, modal_w + 40, modal_h + 40), border_radius=28)
            self.screen.blit(glow_surf, (modal_x - 20, modal_y - 20))
            
            # Modal body
            modal_surface = pygame.Surface((modal_w, modal_h), pygame.SRCALPHA)
            pygame.draw.rect(modal_surface, (25, 25, 50), (0, 0, modal_w, modal_h), border_radius=24)
            pygame.draw.rect(modal_surface, (255, 255, 255, 25), (0, 0, modal_w, modal_h), 
                           width=1, border_radius=24)
            self.screen.blit(modal_surface, (modal_x, modal_y))
            
            # Title with glow
            prompt = FONT_BOLD_MEDIUM.render("Choose a Color", True, COLORS['WHITE'])
            prompt_rect = prompt.get_rect(center=(WINDOW_WIDTH // 2, modal_y + 45))
            self.screen.blit(prompt, prompt_rect)
            
            # Update color button positions
            color_btn_size = 100
            color_spacing = 120
            color_start_x = WINDOW_WIDTH // 2 - (color_spacing * 2 - 20)
            
            for i, btn in enumerate(self.color_buttons):
                btn.rect = pygame.Rect(color_start_x + i * color_spacing, modal_y + 85, color_btn_size, 60)
                btn.render(self.screen)
        
        # Exit button
        self.exit_button.rect = pygame.Rect(WINDOW_WIDTH - 100, 20, 80, 36)
        self.exit_button.render(self.screen)
        
        # Premium turn indicator badge (top right)
        turn_badge_w = 180
        if self.state == GameState.PLAYER_TURN:
            turn_color = COLORS['GOLD']
            turn_label = "Your Turn"
        elif self.state == GameState.RL_PLAYER_TURN:
            turn_color = COLORS['ACCENT_GREEN']
            dots = "." * ((self.animation_offset // 12) % 4)
            turn_label = f"AI Playing{dots}"
        elif self.state == GameState.AI_TURN:
            turn_color = COLORS['ACCENT_PURPLE']
            dots = "." * ((self.animation_offset // 12) % 4)
            turn_label = f"AI Turn{dots}"
        else:
            turn_color = COLORS['GRAY']
            turn_label = ""
        
        if turn_label:
            turn_badge = pygame.Surface((turn_badge_w, 38), pygame.SRCALPHA)
            pygame.draw.rect(turn_badge, (*turn_color[:3], 200), (0, 0, turn_badge_w, 38), border_radius=19)
            pygame.draw.rect(turn_badge, (255, 255, 255, 40), (0, 0, turn_badge_w, 38), width=1, border_radius=19)
            turn_text = FONT_SMALL.render(turn_label, True, COLORS['WHITE'])
            turn_text_rect = turn_text.get_rect(center=(turn_badge_w // 2, 19))
            turn_badge.blit(turn_text, turn_text_rect)
            self.screen.blit(turn_badge, (WINDOW_WIDTH - turn_badge_w - 110, 20))
        
        # Particles
        for particle in self.particles:
            particle.render(self.screen)
        
        # Premium message toast notification with slide animation
        if self.message_timer > 0:
            # Slide-in effect based on timer
            slide_progress = min(1.0, (120 - self.message_timer) / 10) if self.message_timer < 110 else 1.0
            fade_progress = min(1.0, self.message_timer / 30)
            alpha = int(255 * fade_progress)
            
            # Create toast notification
            msg_surface = FONT_REGULAR.render(self.message, True, COLORS['WHITE'])
            toast_w = msg_surface.get_width() + 60
            toast_h = 54
            
            toast = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
            pygame.draw.rect(toast, (25, 25, 50, min(240, alpha)), (0, 0, toast_w, toast_h), border_radius=27)
            pygame.draw.rect(toast, (255, 255, 255, min(30, alpha // 5)), (0, 0, toast_w, toast_h), 
                           width=1, border_radius=27)
            toast.blit(msg_surface, (30, toast_h // 2 - msg_surface.get_height() // 2))
            
            toast_x = WINDOW_WIDTH // 2 - toast_w // 2
            toast_y = 150 - int(20 * (1 - slide_progress))
            self.screen.blit(toast, (toast_x, toast_y))
    
    def render_game_over(self):
        """Render game over screen with premium celebratory design."""
        self.render_background()
        
        # Animated dark overlay with gradient
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Spawn confetti for winner
        if self.winner in ["You", "RL Agent"] and random.random() < 0.15:
            px = random.randint(100, WINDOW_WIDTH - 100)
            color = random.choice([COLORS['GOLD'], COLORS['ACCENT_CYAN'], COLORS['ACCENT_GREEN'], COLORS['WHITE']])
            self.particles.append(Particle(px, -10, color, 'confetti'))
        
        # Premium result panel with glow
        panel_w, panel_h = 540, 380
        panel_x = WINDOW_WIDTH // 2 - panel_w // 2
        panel_y = WINDOW_HEIGHT // 2 - panel_h // 2 - 10
        
        # Outer glow based on winner
        if self.winner in ["You", "RL Agent"]:
            glow_color = COLORS['GOLD'] if self.winner == "You" else COLORS['ACCENT_GREEN']
            glow_surf = pygame.Surface((panel_w + 60, panel_h + 60), pygame.SRCALPHA)
            for i in range(4):
                alpha = 25 - i * 5
                size_offset = i * 10
                pygame.draw.rect(glow_surf, (*glow_color[:3], alpha), 
                               (size_offset, size_offset, panel_w + 60 - size_offset * 2, panel_h + 60 - size_offset * 2), 
                               border_radius=32 - i * 2)
            self.screen.blit(glow_surf, (panel_x - 30, panel_y - 30))
        
        # Panel body
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 20, 45), (0, 0, panel_w, panel_h), border_radius=28)
        pygame.draw.rect(panel, (255, 255, 255, 20), (0, 0, panel_w, panel_h), width=1, border_radius=28)
        self.screen.blit(panel, (panel_x, panel_y))
        
        # Winner announcement
        if self.winner == "You":
            title_text = "Victory!"
            subtitle_text = "Congratulations, You Won!"
            title_color = COLORS['GOLD']
        elif self.winner == "RL Agent":
            title_text = "AI Wins!"
            subtitle_text = "PPO Agent Dominated!"
            title_color = COLORS['ACCENT_GREEN']
        else:
            title_text = "Defeat"
            subtitle_text = "AI Opponent Won This Round"
            title_color = COLORS['ACCENT_ROSE']
        
        # Title with animated glow
        glow_intensity = 0.7 + 0.3 * math.sin(math.radians(self.animation_offset * 3))
        
        # Multi-layer glow
        for i in range(3):
            glow_alpha = int(80 * glow_intensity / (i + 1))
            glow = FONT_BOLD_LARGE.render(title_text, True, (*title_color[:3], glow_alpha))
            offset = (i + 1) * 2
            for ox, oy in [(-offset, 0), (offset, 0), (0, -offset), (0, offset)]:
                glow_rect = glow.get_rect(center=(WINDOW_WIDTH // 2 + ox, panel_y + 75 + oy))
                self.screen.blit(glow, glow_rect)
        
        # Main title
        title = FONT_BOLD_LARGE.render(title_text, True, title_color)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, panel_y + 75))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = FONT_REGULAR.render(subtitle_text, True, COLORS['WHITE'])
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, panel_y + 135))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Stats with icon
        if self.winner in ["You", "RL Agent"]:
            stats_text = f"AI had {len(self.ai_hand)} cards remaining"
        else:
            stats_text = f"You had {len(self.player_hand)} cards remaining"
        stats = FONT_SMALL.render(stats_text, True, COLORS['MUTED'])
        stats_rect = stats.get_rect(center=(WINDOW_WIDTH // 2, panel_y + 180))
        self.screen.blit(stats, stats_rect)
        
        # Elegant divider line
        divider_y = panel_y + 215
        pygame.draw.line(self.screen, (255, 255, 255, 20), 
                        (panel_x + 60, divider_y), (panel_x + panel_w - 60, divider_y), 1)
        
        # Buttons - positioned inside panel
        btn_width = 220
        btn_height = 52
        
        play_again_btn = Button(WINDOW_WIDTH // 2 - btn_width // 2, panel_y + 245, 
                               btn_width, btn_height, "Play Again", COLORS['BUTTON_SUCCESS'])
        play_again_btn.update(pygame.mouse.get_pos())
        play_again_btn.render(self.screen)
        self.play_button.rect = play_again_btn.rect
        self.play_button.text = "Play Again"
        
        menu_btn = Button(WINDOW_WIDTH // 2 - btn_width // 2, panel_y + 310, 
                         btn_width, btn_height, "Main Menu", COLORS['DARK_GRAY'])
        menu_btn.update(pygame.mouse.get_pos())
        menu_btn.render(self.screen)
        self.menu_button.rect = menu_btn.rect
        
        # Exit button
        self.exit_button.rect = pygame.Rect(WINDOW_WIDTH - 100, 20, 80, 36)
        self.exit_button.render(self.screen)
        
        # Particles
        for particle in self.particles:
            particle.render(self.screen)
    
    def render(self):
        """Main render method."""
        if self.state == GameState.MENU:
            self.render_menu()
        elif self.state == GameState.GAME_OVER:
            self.render_game_over()
        else:
            self.render_game()
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    game = UnoGame()
    game.run()
