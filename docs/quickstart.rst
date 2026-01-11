==========
Quickstart
==========

This guide will help you get started with UNO Card Game RL in just a few minutes.

Playing Your First Game
=======================

Play Against AI
---------------

Launch the main GUI and play against trained AI agents:

.. code-block:: bash

    python uno_gui.py

**Controls:**

1. Select a model from the dropdown (e.g., "Best Recurrent PPO")
2. Click **"Play vs AI"** to start
3. Click on playable cards (highlighted) to play them
4. Watch the AI take its turns

Watch AI vs AI
--------------

Observe trained models compete against each other:

.. code-block:: bash

    python uno_gui.py
    # Then click "AI vs AI"

Model Battle Arena
------------------

Compare multiple models in batch battles:

.. code-block:: bash

    python model_battle_gui.py

**Features:**

- Select 2-4 players
- Choose different models for each player
- Run batch evaluations (10-1000 games)
- Export results to CSV

Available Models
================

Our trained models with their approximate win rates:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Model
     - Win Rate
     - Description
   * - **Self-Play Champion** ⭐
     - 70%+
     - Best model, self-play trained
   * - **Best Recurrent PPO**
     - 60%
     - LSTM-based, highest tested
   * - Optimal Recurrent PPO
     - 59%
     - Hyperparameter optimized
   * - SB3 Recurrent PPO
     - 57%
     - Standard SB3 training
   * - Best PPO
     - 53%
     - MLP policy
   * - DQN
     - 48%
     - Value-based
   * - Random Agent
     - 25%
     - Baseline

Running Training
================

Train Your Own Agent
--------------------

Quick training with PPO:

.. code-block:: bash

    python train_rl.py --algorithm ppo --timesteps 100000

Train with Self-Play (Recommended)
----------------------------------

For the best results, use self-play training:

.. code-block:: bash

    python training/train_selfplay.py --mode selfplay --timesteps 1000000

Evaluate a Model
----------------

Compare your trained model against baselines:

.. code-block:: bash

    python compare_models.py --model models/your_model.zip --games 100

Command Reference
=================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Command
     - Description
   * - ``python uno_gui.py``
     - Main game GUI
   * - ``python model_battle_gui.py``
     - Model comparison arena
   * - ``python multiplayer_gui.py``
     - 3-4 player multiplayer
   * - ``python run.py``
     - Quick text-based game
   * - ``python compare_models.py``
     - Batch model evaluation
   * - ``python train_rl.py``
     - Train new model
   * - ``python training/train_selfplay.py``
     - Self-play training

What's Next?
============

- :doc:`gui_guide` - Detailed GUI documentation
- :doc:`training` - Training your own agents
- :doc:`algorithms` - Understanding the RL algorithms
- :doc:`api/index` - API reference for developers
