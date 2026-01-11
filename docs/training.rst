========
Training
========

This guide covers how to train your own UNO RL agents.

Quick Training
==============

Standard PPO Training
---------------------

.. code-block:: bash

    python train_rl.py --algorithm ppo --timesteps 500000

This will train a PPO agent for 500K timesteps and save to ``models/``.

Recurrent PPO Training
----------------------

For better results with LSTM:

.. code-block:: bash

    python train_recurrent_ppo.py --timesteps 1000000

Training Parameters
===================

Command Line Arguments
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Argument
     - Default
     - Description
   * - ``--timesteps``
     - 100000
     - Total training timesteps
   * - ``--algorithm``
     - ppo
     - Algorithm: ppo, dqn, a2c
   * - ``--learning-rate``
     - 3e-4
     - Learning rate
   * - ``--batch-size``
     - 64
     - Batch size for updates
   * - ``--eval-freq``
     - 10000
     - Evaluation frequency
   * - ``--eval-episodes``
     - 100
     - Episodes per evaluation
   * - ``--save-path``
     - models/
     - Where to save model
   * - ``--log-dir``
     - logs/
     - TensorBoard log directory
   * - ``--seed``
     - 42
     - Random seed

Example with Custom Parameters
------------------------------

.. code-block:: bash

    python train_rl.py \
        --algorithm ppo \
        --timesteps 2000000 \
        --learning-rate 1e-4 \
        --batch-size 128 \
        --eval-freq 50000 \
        --seed 123

Training Scripts
================

Available Training Scripts
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Description
   * - ``train_rl.py``
     - General training script (PPO, DQN, A2C)
   * - ``train_sb3.py``
     - Stable-Baselines3 focused training
   * - ``train_recurrent_ppo.py``
     - Standard RecurrentPPO training
   * - ``train_best_recurrent_ppo.py``
     - Optimized RecurrentPPO
   * - ``train_optimal_recurrent_ppo.py``
     - Hyperparameter-tuned RecurrentPPO
   * - ``train_best_ppo.py``
     - Best non-recurrent PPO
   * - ``training/train_selfplay.py``
     - Self-play training (recommended)

Using Config File
-----------------

Modify ``config.py`` for persistent settings:

.. code-block:: python

    training_config = {
        "timesteps": 1000000,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
    }

Monitoring Training
===================

TensorBoard
-----------

View training progress with TensorBoard:

.. code-block:: bash

    tensorboard --logdir logs/

Open http://localhost:6006 in your browser to see:

- Episode rewards
- Episode lengths
- Loss curves
- Learning rate
- Explained variance

Evaluation During Training
--------------------------

Enable periodic evaluation:

.. code-block:: bash

    python train_rl.py --eval-freq 10000 --eval-episodes 100

Results are saved to ``logs/evaluations.npz``.

Checkpointing
=============

Save Checkpoints
----------------

Checkpoints are automatically saved during training:

.. code-block:: python

    from stable_baselines3.common.callbacks import CheckpointCallback
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models/checkpoints/",
        name_prefix="uno_model"
    )

Load from Checkpoint
--------------------

Resume training from a checkpoint:

.. code-block:: python

    from sb3_contrib import RecurrentPPO
    
    model = RecurrentPPO.load("models/checkpoints/uno_model_500000_steps")
    model.learn(total_timesteps=500000)  # Continue training

Best Practices
==============

1. **Start Small**: Begin with 100K steps to verify everything works.

2. **Use RecurrentPPO**: For UNO, LSTM-based models consistently outperform MLP.

3. **Monitor Early**: Check TensorBoard after 10K steps to catch issues.

4. **Save Often**: Use checkpoints every 50K steps.

5. **Evaluate Consistently**: Always evaluate against the same opponents.

6. **Use Self-Play**: For 70%+ win rates, self-play training is essential.

Common Issues
=============

Training Doesn't Converge
-------------------------

- Lower learning rate (try 1e-4 or 1e-5)
- Increase batch size
- Check reward function
- Ensure environment is correct

Slow Training
-------------

- Reduce ``n_steps`` for faster updates
- Use smaller network
- Enable GPU (install PyTorch with CUDA)

Model Overfits
--------------

- Increase entropy coefficient (``ent_coef=0.05``)
- Use self-play training
- Train against diverse opponents

GPU Training
============

Enable GPU training (requires CUDA):

.. code-block:: bash

    # Install PyTorch with CUDA
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    
    # Training automatically uses GPU if available
    python train_rl.py --timesteps 1000000

Check GPU availability:

.. code-block:: python

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
