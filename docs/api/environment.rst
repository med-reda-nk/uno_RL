==================
Environment Module
==================

.. module:: src.state_action_reward
   :synopsis: Gymnasium RL environment for UNO

The environment module provides the RL interface to the UNO game.

UnoEnv Class
============

.. class:: UnoEnv()

   Gymnasium environment for UNO.
   
   This environment wraps the UNO game for reinforcement learning.
   
   .. attribute:: observation_space
      :type: gym.spaces.Box
      
      17-dimensional continuous observation space.
      Shape: (17,), Range: [0, 1]

   .. attribute:: action_space
      :type: gym.spaces.Discrete
      
      9 discrete actions.

   .. method:: reset(seed=None, options=None)
      
      Reset the environment for a new game.
      
      :param seed: Random seed for reproducibility
      :type seed: Optional[int]
      :param options: Additional options (unused)
      :type options: Optional[dict]
      :returns: Initial observation and info dict
      :rtype: Tuple[np.ndarray, dict]

   .. method:: step(action)
      
      Execute an action in the environment.
      
      :param action: Action index (0-8)
      :type action: int
      :returns: (observation, reward, done, truncated, info)
      :rtype: Tuple[np.ndarray, float, bool, bool, dict]

   .. method:: get_valid_actions()
      
      Get list of valid action indices.
      
      :returns: Valid action indices
      :rtype: List[int]

   .. method:: render(mode='human')
      
      Render the current game state.
      
      :param mode: Render mode ('human' or 'ansi')
      :type mode: str

Observation Space
=================

The 17-dimensional observation encodes:

.. list-table::
   :header-rows: 1
   :widths: 15 40 20 25

   * - Index
     - Description
     - Range
     - Encoding
   * - 0-3
     - Open card color
     - [0, 1]
     - One-hot
   * - 4-7
     - Cards per color in hand
     - [0, 1]
     - Normalized count
   * - 8-10
     - Special cards (Skip/Rev/+2)
     - [0, 1]
     - Normalized count
   * - 11-12
     - Wild cards
     - [0, 1]
     - Normalized count
   * - 13-16
     - Playable colors
     - [0, 1]
     - Binary

Action Space
============

.. list-table::
   :header-rows: 1
   :widths: 15 30 55

   * - Index
     - Action
     - Description
   * - 0
     - RED
     - Play any red card from hand
   * - 1
     - GREEN
     - Play any green card from hand
   * - 2
     - BLUE
     - Play any blue card from hand
   * - 3
     - YELLOW
     - Play any yellow card from hand
   * - 4
     - SKIP
     - Play skip card (any color)
   * - 5
     - REVERSE
     - Play reverse card (any color)
   * - 6
     - DRAW2
     - Play draw two card (any color)
   * - 7
     - DRAW4
     - Play wild draw four
   * - 8
     - WILD
     - Play wild card (choose best color)

Reward Structure
================

.. code-block:: python

    def _get_reward(self, done, winner):
        if not done:
            return 0.0
        if winner == 0:  # Agent wins
            return 1.0
        return -1.0  # Agent loses

MultiplayerUnoEnv Class
=======================

.. class:: MultiplayerUnoEnv(num_players=4)

   Extended environment for 2-4 players.
   
   :param num_players: Number of players (2-4)
   :type num_players: int
   
   Extends observation to 25 dimensions to include opponent hand sizes.

   .. attribute:: observation_space
      :type: gym.spaces.Box
      
      25-dimensional observation space for multiplayer.

   .. attribute:: direction
      :type: int
      
      Current turn direction (1=clockwise, -1=counter-clockwise)

   .. attribute:: current_player
      :type: int
      
      Index of current player (0 to num_players-1)

Example Usage
=============

Basic Usage
-----------

.. code-block:: python

    from src.state_action_reward import UnoEnv
    import numpy as np
    
    # Create environment
    env = UnoEnv()
    
    # Reset for new game
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    
    # Game loop
    done = False
    total_reward = 0
    
    while not done:
        # Get valid actions
        valid_actions = env.get_valid_actions()
        
        # Random policy
        action = np.random.choice(valid_actions)
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    
    print(f"Game over! Total reward: {total_reward}")

With Stable-Baselines3
----------------------

.. code-block:: python

    from src.state_action_reward import UnoEnv
    from sb3_contrib import RecurrentPPO
    
    # Create environment
    env = UnoEnv()
    
    # Train model
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    
    # Evaluate
    obs, _ = env.reset()
    lstm_state = None
    episode_reward = 0
    
    while True:
        action, lstm_state = model.predict(obs, state=lstm_state, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    
    print(f"Episode reward: {episode_reward}")

Multiplayer Usage
-----------------

.. code-block:: python

    from src.multiplayer_env import MultiplayerUnoEnv
    
    # 4-player game
    env = MultiplayerUnoEnv(num_players=4)
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")  # (25,)
    print(f"Current player: {env.current_player}")
    print(f"Turn direction: {env.direction}")
