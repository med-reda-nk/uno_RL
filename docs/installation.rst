============
Installation
============

This guide covers how to install the UNO Card Game RL project on your system.

Requirements
============

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

System Dependencies
-------------------

**Windows**
    No additional system dependencies required.

**Linux**
    You may need SDL libraries for Pygame:
    
    .. code-block:: bash
    
        sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev

**macOS**
    Install SDL via Homebrew:
    
    .. code-block:: bash
    
        brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf

Standard Installation
=====================

1. Clone the Repository
-----------------------

.. code-block:: bash

    git clone https://github.com/user/uno-card-game-rl.git
    cd uno-card-game-rl

2. Create Virtual Environment (Recommended)
-------------------------------------------

.. code-block:: bash

    # Create virtual environment
    python -m venv venv
    
    # Activate (Windows)
    venv\Scripts\activate
    
    # Activate (Linux/macOS)
    source venv/bin/activate

3. Install Dependencies
-----------------------

.. code-block:: bash

    pip install -r requirements.txt

Development Installation
========================

For development, install with editable mode:

.. code-block:: bash

    pip install -e .
    pip install -r requirements-dev.txt

Required Packages
=================

Core Dependencies
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Purpose
   * - gymnasium
     - ≥0.29.0
     - RL environment framework
   * - stable-baselines3
     - ≥2.0.0
     - RL algorithms (PPO, DQN, A2C)
   * - sb3-contrib
     - ≥2.0.0
     - RecurrentPPO (LSTM policies)
   * - pygame
     - ≥2.5.0
     - GUI rendering
   * - numpy
     - ≥1.24.0
     - Numerical operations
   * - pandas
     - ≥2.0.0
     - Data analysis
   * - torch
     - ≥2.0.0
     - Deep learning backend

Optional Dependencies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Package
     - Version
     - Purpose
   * - tensorboard
     - ≥2.14.0
     - Training visualization
   * - matplotlib
     - ≥3.7.0
     - Plotting results
   * - seaborn
     - ≥0.12.0
     - Statistical plots

Verifying Installation
======================

Run the following to verify everything is installed correctly:

.. code-block:: bash

    # Test basic imports
    python -c "from src.game import UnoGame; print('✓ Game engine OK')"
    
    # Test GUI
    python -c "import pygame; pygame.init(); print('✓ Pygame OK')"
    
    # Test RL libraries
    python -c "from sb3_contrib import RecurrentPPO; print('✓ RecurrentPPO OK')"

Troubleshooting
===============

Common Issues
-------------

**ImportError: No module named 'pygame'**
    Run ``pip install pygame`` and ensure your virtual environment is activated.

**CUDA not available**
    Training will work on CPU. For GPU acceleration, install PyTorch with CUDA support:
    
    .. code-block:: bash
    
        pip install torch --index-url https://download.pytorch.org/whl/cu118

**Model file not found**
    Ensure you're running from the project root directory where the ``models/`` folder exists.

Getting Help
------------

If you encounter issues:

1. Check the :doc:`quickstart` guide
2. Search existing GitHub issues
3. Open a new issue with your error message and system info
