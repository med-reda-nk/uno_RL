==========
Algorithms
==========

This page explains the Reinforcement Learning algorithms used in this project.

Overview
========

We explored several RL algorithms for learning to play UNO:

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Algorithm
     - Type
     - Win Rate
     - Best For
   * - Recurrent PPO
     - Policy
     - 60%+
     - Partial observability (recommended)
   * - PPO
     - Policy
     - 53%
     - Stable training
   * - DQN
     - Value
     - 48%
     - Discrete actions
   * - A2C
     - Policy
     - 45%
     - Fast training
   * - Q-Learning
     - Tabular
     - 35%
     - Small state spaces

Q-Learning
==========

The simplest algorithm we tested. Uses a table to store Q-values for each state-action pair.

**Update Rule:**

.. math::

    Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]

Where:
- :math:`\alpha` is the learning rate
- :math:`\gamma` is the discount factor
- :math:`r` is the immediate reward

**Limitations:**
- State space too large for tables
- No generalization across similar states
- Only achieves ~35% win rate

Deep Q-Network (DQN)
====================

Uses neural networks to approximate Q-values, enabling generalization.

**Architecture:**

.. code-block:: text

    Input (17) → Dense(256) → ReLU → Dense(256) → ReLU → Output (9)

**Key Techniques:**

1. **Experience Replay**: Store transitions in buffer, sample randomly
2. **Target Network**: Separate network for stable targets
3. **ε-greedy Exploration**: Random actions with probability ε

**Configuration:**

.. code-block:: python

    DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        target_update_interval=1000
    )

Proximal Policy Optimization (PPO)
==================================

A policy gradient method with clipped objective for stable training.

**Objective:**

.. math::

    L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) \right]

Where :math:`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}` is the probability ratio.

**Benefits:**

- Stable training
- Sample efficient
- Works well with neural networks

**Configuration:**

.. code-block:: python

    PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )

Recurrent PPO (LSTM)
====================

Our **best performing** algorithm! Uses LSTM networks to maintain memory of past observations.

**Why LSTM for UNO?**

UNO is a **partially observable** game:
- You can't see opponent's cards
- Cards in the deck are hidden
- Past plays contain useful information

LSTM maintains a **hidden state** that accumulates information over time.

**Architecture:**

.. code-block:: text

    Input (17) → LSTM(256) → Dense(256) → Dense(128) → Policy(9) + Value(1)

**Key Insight:**

The LSTM learns to:
1. Track which cards have been played
2. Estimate remaining cards in the deck
3. Infer opponent's hand composition
4. Remember game patterns

**Configuration:**

.. code-block:: python

    RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=True
        )
    )

Algorithm Comparison
====================

Training Efficiency
-------------------

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Timesteps to 50% WR
     - GPU Memory
     - Training Time (1M steps)
   * - Q-Learning
     - Never reached
     - N/A
     - 10 min
   * - DQN
     - 800K
     - ~2GB
     - 30 min
   * - PPO
     - 500K
     - ~1GB
     - 20 min
   * - Recurrent PPO
     - 300K
     - ~3GB
     - 45 min

Recommendations
---------------

**For beginners:** Start with PPO - it's stable and easy to tune.

**For best results:** Use Recurrent PPO - the memory is essential for UNO.

**For research:** Try different architectures with the RecurrentPPO base.

Hyperparameter Tuning
=====================

Key parameters to tune for UNO:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Range
     - Effect
   * - learning_rate
     - 1e-5 to 1e-3
     - Higher = faster but unstable
   * - n_steps
     - 64 to 2048
     - Higher = more stable gradients
   * - batch_size
     - 32 to 256
     - Must divide n_steps evenly
   * - gamma
     - 0.9 to 0.999
     - Higher = more farsighted
   * - ent_coef
     - 0.0 to 0.1
     - Higher = more exploration

Our best configuration is in ``config.py``.
