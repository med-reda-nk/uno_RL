=====
Usage
=====

This guide covers all the ways to use the UNO Card Game RL system.

Game Modes
==========

Human vs AI
-----------

The primary game mode where you play against a trained AI agent.

.. code-block:: bash

    python uno_gui.py

**Gameplay:**

1. Launch the GUI
2. Select your preferred AI model from the dropdown
3. Click "Play vs AI"
4. Wait for cards to be dealt
5. Click on highlighted (playable) cards to play them
6. If no playable card, click the deck to draw
7. First to empty their hand wins!

AI vs AI (Spectator Mode)
-------------------------

Watch two AI agents battle each other:

.. code-block:: bash

    python uno_gui.py
    # Click "AI vs AI"

This mode is useful for:

- Observing AI strategies
- Debugging agent behavior  
- Entertainment

Model Battle Arena
------------------

The battle arena allows you to:

- Compare multiple models head-to-head
- Run batch evaluations (10-1000 games)
- Track statistics (wins, win rates, average turns)
- Export results to CSV

.. code-block:: bash

    python model_battle_gui.py

**Features:**

- 2-4 player support
- Select different models for each player
- Real-time statistics
- Progress bar for batch runs

Multiplayer Mode
----------------

Play with 3-4 AI agents simultaneously:

.. code-block:: bash

    python multiplayer_gui.py --players 4

Command-Line Interface
======================

Quick Game
----------

For a fast text-based game:

.. code-block:: bash

    python run.py

Model Comparison
----------------

Compare models programmatically:

.. code-block:: bash

    python compare_models.py --games 100

Configuration
=============

Environment Variables
---------------------

Set these environment variables to customize behavior:

.. code-block:: bash

    # Disable GPU (CPU only)
    export CUDA_VISIBLE_DEVICES=""
    
    # Set random seed
    export UNO_SEED=42

Config File
-----------

Edit ``config.py`` to modify:

- Model paths
- Training hyperparameters
- Game settings

.. code-block:: python

    # config.py
    sb3_models = {
        "selfplay_champion": {
            "name": "Self-Play Champion",
            "path": "models/selfplay_champion.zip",
            "win_rate": 0.70
        },
        # ... other models
    }

Keyboard Shortcuts
==================

GUI Controls
------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Key
     - Action
     - Context
   * - ESC
     - Return to menu
     - During game
   * - Space
     - Draw card (if no playable)
     - Your turn
   * - R
     - Restart game
     - Game over
   * - Q
     - Quit
     - Any screen

Mouse Controls
--------------

- **Left Click**: Select cards, buttons
- **Hover**: View card details
- **Scroll**: (Battle arena) Scroll through history

Output Files
============

The system generates several output files:

Comparison Results
------------------

Located in ``comparison_results/``:

- ``trained_models_YYYYMMDD_HHMMSS.csv`` - Batch evaluation results
- ``existing_models_YYYYMMDD_HHMMSS.csv`` - Pre-trained model comparison

Training Logs
-------------

Located in ``logs/``:

- ``evaluations.npz`` - Evaluation metrics during training
- ``*/events.out.tfevents.*`` - TensorBoard logs

View with TensorBoard:

.. code-block:: bash

    tensorboard --logdir logs/

Assets
------

Located in ``assets/``:

- ``*.csv`` - Historical training data
- ``*.png`` - Training plots (if generated)
