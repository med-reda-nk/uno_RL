=============
API Reference
=============

This section provides detailed API documentation for all modules.

.. toctree::
   :maxdepth: 2

   game
   cards
   environment
   agents

Quick Links
===========

Core Modules
------------

- :doc:`game` - UNO game engine
- :doc:`cards` - Card and deck classes
- :doc:`environment` - Gymnasium RL environment
- :doc:`agents` - Agent implementations

Usage Example
=============

.. code-block:: python

    from src.game import UnoGame
    from src.state_action_reward import UnoEnv
    from src.agents import SB3Agent
    
    # Create environment
    env = UnoEnv()
    
    # Load agent
    agent = SB3Agent("models/selfplay_champion.zip")
    
    # Play a game
    obs = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(obs, env.get_valid_actions())
        obs, reward, done, truncated, info = env.step(action)
    
    print(f"Game result: {'Win' if reward > 0 else 'Loss'}")
