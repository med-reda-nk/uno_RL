============
Architecture
============

This document describes the overall architecture of the UNO Card Game RL project.

Project Structure
=================

.. code-block:: text

    uno-card-game-rl/
    ├── src/                    # Core source code
    │   ├── __init__.py
    │   ├── game.py            # UNO game engine
    │   ├── cards.py           # Card definitions
    │   ├── players.py         # Player classes
    │   ├── turn.py            # Turn management
    │   ├── agents.py          # RL agent wrappers
    │   ├── dqn_agent.py       # DQN implementation
    │   ├── sb3_agent.py       # Stable-Baselines3 agents
    │   ├── state_action_reward.py  # RL interface
    │   ├── utils.py           # Utilities
    │   └── multiplayer_env.py # Multiplayer environment
    │
    ├── training/              # Training scripts
    │   └── train_selfplay.py  # Self-play training
    │
    ├── models/                # Saved models
    │   ├── selfplay_champion.zip
    │   ├── best_recurrent_ppo.zip
    │   └── ...
    │
    ├── docs/                  # Documentation
    │   ├── conf.py
    │   ├── index.rst
    │   └── ...
    │
    ├── logs/                  # Training logs
    ├── assets/                # Data files
    ├── comparison_results/    # Evaluation results
    │
    ├── uno_gui.py            # Main game GUI
    ├── model_battle_gui.py   # Battle arena GUI
    ├── multiplayer_gui.py    # Multiplayer GUI
    │
    ├── train_rl.py           # General training script
    ├── compare_models.py     # Model comparison
    ├── config.py             # Configuration
    ├── run.py                # Quick run script
    └── requirements.txt      # Dependencies

Core Components
===============

Game Engine
-----------

The game engine (``src/game.py``) handles all UNO game logic:

.. code-block:: python

    class UnoGame:
        """
        Main UNO game controller.
        
        Manages:
        - Deck and discard pile
        - Player hands
        - Turn order
        - Win conditions
        """
        
        def play_card(self, player, card):
            """Execute a card play."""
            
        def draw_card(self, player):
            """Player draws from deck."""
            
        def get_winner(self):
            """Return winner if game over, else None."""

Card System
-----------

Cards (``src/cards.py``) are represented as:

.. code-block:: python

    @dataclass
    class Card:
        color: str      # 'red', 'green', 'blue', 'yellow', 'wild'
        value: str      # '0'-'9', 'skip', 'reverse', 'draw2', 'wild', 'draw4'
        
    class Deck:
        """Standard 108-card UNO deck."""
        
        def shuffle(self): ...
        def draw(self): ...

RL Environment
--------------

The Gymnasium environment (``src/state_action_reward.py``):

.. code-block:: python

    class UnoEnv(gym.Env):
        """
        UNO as a Gymnasium environment.
        
        Observation: 17-dim vector
        Actions: 9 discrete actions
        Reward: +1 win, -1 loss, 0 ongoing
        """
        
        observation_space = spaces.Box(low=0, high=1, shape=(17,))
        action_space = spaces.Discrete(9)
        
        def step(self, action):
            # Execute action, return (obs, reward, done, truncated, info)
            
        def reset(self):
            # Start new game, return initial observation

Agent Wrappers
--------------

Agents (``src/agents.py``) provide unified interface:

.. code-block:: python

    class RLAgent:
        """Base class for all RL agents."""
        
        def select_action(self, obs, valid_actions):
            """Return action index."""
            
    class SB3Agent(RLAgent):
        """Wrapper for Stable-Baselines3 models."""
        
        def __init__(self, model_path):
            self.model = RecurrentPPO.load(model_path)
            
        def select_action(self, obs, valid_actions):
            action, self.state = self.model.predict(obs, state=self.state)
            return action

Data Flow
=========

Training Flow
-------------

.. code-block:: text

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ Environment │ ←── │   Agent     │ ←── │   Model     │
    │  (UnoEnv)   │     │ (SB3Agent)  │     │ (RecPPO)    │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           │    observation    │                   │
           │ ─────────────────→│   observation     │
           │                   │ ─────────────────→│
           │                   │                   │
           │                   │      action       │
           │      action       │ ←─────────────────│
           │ ←─────────────────│                   │
           │                   │                   │
           │ reward, done      │                   │
           │ ─────────────────→│   experience      │
           │                   │ ─────────────────→│
           └───────────────────┴───────────────────┘

Inference Flow
--------------

.. code-block:: text

    User Input → GUI → Agent.select_action() → Environment.step() → GUI Update

State Representation
====================

17-Dimensional Observation
--------------------------

.. list-table::
   :header-rows: 1

   * - Index
     - Feature
     - Range
   * - 0-3
     - Open card color (one-hot)
     - [0, 1]
   * - 4-7
     - Number cards per color
     - [0, 1] normalized
   * - 8-10
     - Special cards (Skip/Rev/+2)
     - [0, 1] normalized
   * - 11-12
     - Wild cards count
     - [0, 1] normalized
   * - 13-16
     - Playable colors
     - [0, 1]

Action Encoding
---------------

.. list-table::
   :header-rows: 1

   * - Index
     - Action
     - Description
   * - 0
     - RED
     - Play any red card
   * - 1
     - GREEN
     - Play any green card
   * - 2
     - BLUE
     - Play any blue card
   * - 3
     - YELLOW
     - Play any yellow card
   * - 4
     - SKIP
     - Play skip card
   * - 5
     - REVERSE
     - Play reverse card
   * - 6
     - DRAW2
     - Play +2 card
   * - 7
     - DRAW4
     - Play wild +4
   * - 8
     - WILD
     - Play wild card

Reward Structure
----------------

.. code-block:: python

    def get_reward(self, done, winner):
        if not done:
            return 0.0  # Game ongoing
        if winner == self.agent_player:
            return 1.0  # Win
        return -1.0  # Loss

Neural Network Architecture
===========================

RecurrentPPO Network
--------------------

.. code-block:: text

    ┌──────────────────┐
    │  Observation     │
    │    (17 dim)      │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │      LSTM        │
    │   (256 hidden)   │
    │                  │
    │  h_t = LSTM(     │
    │    x_t, h_{t-1}) │
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │    MLP Layers    │
    │  256 → 128 → 64  │
    │     (ReLU)       │
    └────────┬─────────┘
             │
       ┌─────┴─────┐
       │           │
    ┌──▼───┐   ┌───▼──┐
    │Policy│   │Value │
    │(9dim)│   │(1dim)│
    └──────┘   └──────┘

Why LSTM?
---------

LSTM enables:

1. **Memory**: Track cards played earlier in game
2. **Inference**: Deduce opponent's hand from play history
3. **Strategy**: Remember opponent patterns
4. **Context**: Understand game state evolution

GUI Architecture
================

All GUIs use Pygame with a similar structure:

.. code-block:: python

    class GameGUI:
        def __init__(self):
            pygame.init()
            self.screen = pygame.display.set_mode((1280, 720))
            self.clock = pygame.time.Clock()
            
        def run(self):
            while self.running:
                self.handle_events()
                self.update()
                self.draw()
                pygame.display.flip()
                self.clock.tick(60)

Component Hierarchy
-------------------

.. code-block:: text

    GameGUI
    ├── MenuScreen
    │   ├── ModelSelector
    │   └── Buttons
    ├── GameScreen
    │   ├── CardRenderer
    │   ├── DeckDisplay
    │   └── ActionLog
    └── EndScreen

Extension Points
================

Adding New Algorithms
---------------------

1. Create agent wrapper in ``src/agents.py``
2. Add training script
3. Register in ``config.py``
4. Add to GUI model list

Adding New Features
-------------------

1. Extend ``UnoEnv`` observation/action space
2. Update ``UnoGame`` logic
3. Modify GUI rendering
4. Update documentation

Testing
=======

Unit tests are in ``tests/``:

.. code-block:: bash

    pytest tests/ -v

Test coverage includes:

- Card mechanics
- Game rules
- Environment interface
- Agent behavior
