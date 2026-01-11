==========
GUI Guide
==========

This guide covers all graphical interfaces in the UNO Card Game RL project.

Main Game GUI
=============

Launch with:

.. code-block:: bash

    python uno_gui.py

Menu Screen
-----------

The main menu offers:

- **Model Selector**: Dropdown to choose AI opponent
- **Play vs AI**: Human vs selected AI model
- **AI vs AI**: Watch two AIs compete
- **Multiplayer (3-4)**: Launch multiplayer mode
- **Battle Arena**: Open model comparison GUI
- **Quit**: Exit application

Model Selection
---------------

Available models in the dropdown:

1. **[NEW] Self-Play Champion** (70%+) - Recommended
2. **[BEST] Best Recurrent PPO** (60%)
3. **Optimal Recurrent PPO** (59%)
4. **SB3 Recurrent PPO** (57%)
5. **Best PPO** (53%)
6. **DQN Agent** (48%)
7. **A2C Agent** (45%)

Gameplay Controls
-----------------

**Mouse:**

- Click playable cards (green highlight) to play
- Click deck to draw when no playable cards
- Click UNO button when you have one card left

**Keyboard:**

- ``ESC`` - Return to menu
- ``R`` - Restart game
- ``Space`` - Draw card (alternative)

Game Display
------------

.. code-block:: text

    ┌─────────────────────────────────────┐
    │          OPPONENT CARDS             │
    │         (face down, count shown)    │
    ├─────────────────────────────────────┤
    │                                     │
    │     [DECK]        [DISCARD PILE]    │
    │                                     │
    ├─────────────────────────────────────┤
    │           YOUR CARDS                │
    │     (face up, clickable)            │
    │         [UNO BUTTON]                │
    └─────────────────────────────────────┘

Model Battle GUI
================

Launch with:

.. code-block:: bash

    python model_battle_gui.py

Features
--------

- **2-4 Player Support**: Battle with multiple AIs
- **Model Selectors**: Choose models for each player
- **Batch Evaluation**: Run 10-1000 games
- **Statistics**: Win rates, average turns, etc.
- **CSV Export**: Save results for analysis

Player Count Selection
----------------------

Click the player count buttons:

- **2P**: Standard head-to-head
- **3P**: Three-way battle
- **4P**: Full table competition

Running Battles
---------------

1. Select number of players (2P/3P/4P)
2. Choose a model for each player
3. Enter number of games (10-1000)
4. Click "Start Battle"
5. Watch progress bar
6. Review statistics
7. Optionally export to CSV

Statistics Displayed
--------------------

After each batch:

- Games won per model
- Win percentage
- Average game length
- Total games played

Multiplayer GUI
===============

Launch with:

.. code-block:: bash

    python multiplayer_gui.py
    # Or with specific player count:
    python multiplayer_gui.py --players 4

Features
--------

- **3-4 Players**: Extended game rules
- **Turn Direction**: Reverse card changes direction
- **Skip Mechanics**: Properly skips next player
- **Circular Table**: Visual player arrangement

Player Arrangement
------------------

.. code-block:: text

              Player 2 (AI)
                   │
    Player 3 (AI) ─┼─ Player 1 (Human)
                   │
              Player 4 (AI)

Design Features
===============

Glassmorphism Style
-------------------

All GUIs feature a modern "glassmorphism" design:

- Frosted glass effect backgrounds
- Subtle gradients
- Soft shadows
- Rounded corners

Color Scheme
------------

UNO card colors:

- **Red**: #EF4444
- **Green**: #10B981
- **Blue**: #3B82F6
- **Yellow**: #FACC15
- **Wild**: #6366F1

Card Rendering
--------------

Cards are dynamically rendered with:

- Appropriate color fills
- Number/action symbols
- Rounded corners
- Highlight effects for playable cards

Customization
=============

Changing Window Size
--------------------

Edit the GUI files:

.. code-block:: python

    # In uno_gui.py
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720

Adding New Models
-----------------

Add models to the discovery list:

.. code-block:: python

    # In uno_gui.py
    known_models = {
        "your_model.zip": "Your Model Name",
        # ... existing models
    }

Troubleshooting
===============

GUI Won't Launch
----------------

- Ensure Pygame is installed: ``pip install pygame``
- Check Python version (3.8+)
- Verify display is available

Cards Not Clickable
-------------------

- Only playable cards are highlighted/clickable
- If no cards playable, click the deck to draw

Model Not Loading
-----------------

- Verify model file exists in ``models/`` directory
- Check file extension is ``.zip``
- Ensure sb3-contrib is installed

Performance Issues
------------------

- Close other applications
- Reduce window size
- Disable background animations (edit source)

Screenshots
===========

*Note: Actual screenshots would be included here in a real documentation build.*

Main Menu
---------

The main menu features:
- Project title with UNO-styled text
- Model selector dropdown
- Four main buttons
- Modern gradient background

Gameplay
--------

During gameplay you'll see:
- Opponent's cards (face down)
- Central play area with discard pile
- Your hand with playable cards highlighted
- Turn indicator and action log

Battle Arena
------------

The battle arena displays:
- Player count selector
- Multiple model dropdowns
- Game count input
- Real-time statistics panel
- Progress bar during evaluation
