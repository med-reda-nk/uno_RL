=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_.

[1.0.0] - 2026-01-XX
====================

Added
-----

- **Self-Play Training**: New training framework for achieving 70%+ win rates
  
  - Population-based training with opponent pool
  - Curriculum learning from random to self-play
  - Checkpoint management for diverse opponents

- **Multiplayer Support**: Extended game to 2-4 players
  
  - MultiplayerUnoEnv with proper turn direction
  - 25-dimensional observation including opponent hand sizes
  - Skip and Reverse mechanics for 3-4 players

- **Model Battle Arena**: New GUI for comparing models
  
  - 2-4 player battle support
  - Batch evaluation (10-1000 games)
  - CSV export functionality
  - Multiple model selectors

- **GUI Improvements**
  
  - ModelSelector dropdown in main menu
  - Model discovery from filesystem
  - Multiplayer launcher button
  - Glassmorphism design updates

- **Documentation**
  
  - Complete ReadTheDocs documentation
  - LaTeX report with methodology
  - Presentation slides (Beamer)
  - API reference documentation

Changed
-------

- Updated ``uno_gui.py`` with model selection and multiplayer button
- Extended ``model_battle_gui.py`` for 2-4 players
- Added Self-Play Champion to all comparison scripts
- Improved README with model performance table

Fixed
-----

- Fixed ``gymnasium`` import in train_selfplay.py
- Corrected action masking for invalid plays
- Fixed card rendering for special cards

[0.2.0] - 2025-12-XX
====================

Added
-----

- **Recurrent PPO Implementation**
  
  - LSTM-based policies for partial observability
  - 60% win rate achievement
  - Multiple training configurations

- **Training Scripts**
  
  - train_recurrent_ppo.py
  - train_best_recurrent_ppo.py
  - train_optimal_recurrent_ppo.py

- **Model Comparison**
  
  - compare_models.py for batch evaluation
  - CSV result export
  - TensorBoard logging

Changed
-------

- Switched from gym to gymnasium
- Updated stable-baselines3 to v2.0+
- Improved reward shaping

[0.1.0] - 2025-11-XX
====================

Added
-----

- **Core Game Engine**
  
  - Complete UNO rules implementation
  - Card and Deck classes
  - Game state management

- **Basic RL Environment**
  
  - Gymnasium-compatible UnoEnv
  - 17-dimensional observation
  - 9 discrete actions

- **Initial Agents**
  
  - Q-Learning (tabular)
  - DQN agent
  - PPO and A2C via stable-baselines3

- **Main GUI**
  
  - Pygame-based interface
  - Human vs AI mode
  - AI vs AI spectator mode

- **Training Infrastructure**
  
  - train_rl.py general trainer
  - train_sb3.py for SB3 models
  - Evaluation callbacks
