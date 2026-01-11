==================
Self-Play Training
==================

Self-play is our recommended training method for achieving the highest win rates (70%+).

What is Self-Play?
==================

In self-play training, the agent plays against copies of itself (or previous versions). This creates an ever-improving opponent, preventing overfitting to fixed strategies.

**Benefits:**

- No need for hand-crafted opponents
- Discovers novel strategies
- Robust to different play styles
- Continuously improves
- Prevents exploitation of fixed patterns

How It Works
============

Our self-play implementation uses several techniques:

1. Population-Based Training
----------------------------

Maintain a pool of agents at different skill levels:

.. code-block:: text

    Population Pool
    ├── Current Agent (training)
    ├── Checkpoint @ 100K steps
    ├── Checkpoint @ 200K steps
    ├── Checkpoint @ 500K steps
    └── Best Agent (highest eval score)

2. Opponent Sampling
--------------------

Sample opponents from the pool with probabilities:

.. code-block:: python

    opponent_weights = {
        "current": 0.3,      # Train against itself
        "recent": 0.4,       # Recent checkpoints
        "best": 0.2,         # Best historical
        "random": 0.1        # Maintain exploration
    }

3. Curriculum Learning
----------------------

Gradually increase difficulty:

.. code-block:: text

    Phase 1 (0-100K):   vs Random
    Phase 2 (100K-500K): vs Mix(Random, Self)
    Phase 3 (500K+):     vs Self + Population

Running Self-Play Training
==========================

Basic Usage
-----------

.. code-block:: bash

    python training/train_selfplay.py --mode selfplay --timesteps 1000000

Advanced Options
----------------

.. code-block:: bash

    python training/train_selfplay.py \
        --mode selfplay \
        --timesteps 2000000 \
        --checkpoint-freq 100000 \
        --population-size 5 \
        --learning-rate 1e-4 \
        --eval-episodes 200

Command Line Arguments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--mode``
     - selfplay
     - Training mode: selfplay, curriculum, mixed
   * - ``--timesteps``
     - 1000000
     - Total training timesteps
   * - ``--checkpoint-freq``
     - 100000
     - Save checkpoint frequency
   * - ``--population-size``
     - 5
     - Max opponents in pool
   * - ``--learning-rate``
     - 3e-4
     - Learning rate
   * - ``--eval-episodes``
     - 100
     - Evaluation episodes

Training Modes
==============

Selfplay Mode
-------------

Pure self-play against population:

.. code-block:: bash

    python training/train_selfplay.py --mode selfplay

Curriculum Mode
---------------

Gradual difficulty increase:

.. code-block:: bash

    python training/train_selfplay.py --mode curriculum

Mixed Mode
----------

Combination of techniques:

.. code-block:: bash

    python training/train_selfplay.py --mode mixed

Implementation Details
======================

The self-play environment wrapper:

.. code-block:: python

    class SelfPlayEnv(gym.Wrapper):
        def __init__(self, env, opponent_pool):
            super().__init__(env)
            self.opponent_pool = opponent_pool
            self.current_opponent = None
            
        def reset(self):
            # Sample new opponent each episode
            self.current_opponent = self._sample_opponent()
            return super().reset()
            
        def _sample_opponent(self):
            weights = [0.3, 0.4, 0.2, 0.1]  # current, recent, best, random
            return random.choices(self.opponent_pool, weights=weights)[0]

Monitoring Progress
===================

Track these metrics during self-play:

.. list-table::
   :header-rows: 1

   * - Metric
     - Good Sign
     - Bad Sign
   * - Win rate vs Random
     - > 60%
     - < 50%
   * - Win rate vs Self
     - 45-55%
     - < 30% or > 70%
   * - Episode length
     - Decreasing
     - Increasing
   * - Reward
     - Increasing
     - Flat or decreasing

Expected Results
================

Training Timeline
-----------------

.. code-block:: text

    Timesteps     Win Rate (vs Random)
    ---------     -------------------
    100K          40-45%
    250K          50-55%
    500K          55-60%
    1M            60-65%
    2M            65-70%
    5M            70%+

Final Performance
-----------------

After 2M timesteps of self-play:

- **vs Random**: 70%+ win rate
- **vs PPO**: 60%+ win rate
- **vs DQN**: 65%+ win rate

Tips for Best Results
=====================

1. **Long Training**: Self-play benefits from extended training (2M+ steps)

2. **Large Population**: Use 5-10 agents in the population

3. **Regular Checkpoints**: Save every 100K steps for diversity

4. **Evaluation**: Test against fixed baselines periodically

5. **Patience**: Early training may show unstable metrics

6. **Resources**: Self-play uses more memory (multiple models loaded)

Troubleshooting
===============

Win Rate Not Improving
----------------------

- Increase population diversity
- Add more random agents to pool
- Lower learning rate

Training Unstable
-----------------

- Reduce opponent sampling frequency
- Increase batch size
- Use smaller population

Out of Memory
-------------

- Reduce population size
- Use smaller LSTM hidden size
- Clear old checkpoints

Using the Trained Model
=======================

After training, the champion model is saved to:

.. code-block:: text

    models/selfplay_champion.zip

Use it in the GUI or evaluation:

.. code-block:: bash

    python uno_gui.py  # Select "Self-Play Champion" from dropdown

Or load programmatically:

.. code-block:: python

    from sb3_contrib import RecurrentPPO
    
    model = RecurrentPPO.load("models/selfplay_champion.zip")
