===============================================
UNO Card Game with Reinforcement Learning
===============================================

Welcome to the documentation for the **UNO Card Game RL** project! This project implements a complete UNO card game with multiple Reinforcement Learning agents.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: LICENSE

Overview
========

This project explores the application of Reinforcement Learning to the classic UNO card game. Our agents achieve **60%+ win rates** against random opponents, with our Self-Play Champion targeting **70%+** win rates.

Key Features
------------

- 🃏 **Complete UNO Engine**: Full game rules including special cards
- 🤖 **Multiple RL Agents**: Q-Learning, DQN, PPO, Recurrent PPO
- 🎮 **Modern GUIs**: Pygame-based interfaces with glassmorphism design
- 👥 **Multiplayer Support**: 2-4 player battles
- 📊 **Model Comparison**: Evaluate and compare different agents
- 🔄 **Self-Play Training**: Train agents against themselves

Quick Start
-----------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/user/uno-card-game-rl.git
   cd uno-card-game-rl

   # Install dependencies
   pip install -r requirements.txt

   # Play against AI
   python uno_gui.py

   # Watch AI battles
   python model_battle_gui.py

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage
   gui_guide
   multiplayer

.. toctree::
   :maxdepth: 2
   :caption: Training

   training
   algorithms
   self_play

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   api/index
   contributing

.. toctree::
   :maxdepth: 1
   :caption: Reference

   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
