===========
Multiplayer
===========

This guide covers the multiplayer features supporting 3-4 player games.

Overview
========

While standard UNO is often played with 2 players, our implementation supports 2-4 players for more dynamic gameplay.

Key Differences from 2-Player
-----------------------------

- **Turn Direction**: Reverse card actually reverses turn order
- **Skip Mechanics**: Skip affects the immediate next player
- **Draw Effects**: +2 and +4 target the next player in turn order
- **Strategy**: Must consider multiple opponents

Multiplayer Environment
=======================

The multiplayer environment extends the base UNO environment:

.. code-block:: python

    from src.multiplayer_env import MultiplayerUnoEnv
    
    # Create 4-player game
    env = MultiplayerUnoEnv(num_players=4)
    
    # Reset returns observation for player 0
    obs = env.reset()
    
    # Step returns (obs, reward, done, truncated, info)
    obs, reward, done, truncated, info = env.step(action)

Observation Space
-----------------

Extended 25-dimensional observation:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Feature
     - Dimensions
     - Description
   * - Open card color
     - 4
     - One-hot encoded
   * - Number cards per color
     - 4
     - Cards in hand
   * - Special cards
     - 3
     - Skip, Reverse, +2
   * - Wild cards
     - 2
     - Wild, Wild +4
   * - Playable colors
     - 4
     - Which colors can be played
   * - Opponent 1 hand size
     - 4
     - Bucketed (1-3, 4-6, 7+, UNO)
   * - Opponent 2 hand size
     - 4
     - Same buckets

Action Space
------------

Same 9 actions as 2-player:

0. Play Red card
1. Play Green card  
2. Play Blue card
3. Play Yellow card
4. Play Skip
5. Play Reverse
6. Play +2
7. Play Wild +4
8. Play Wild (choose color)

Turn Direction
==============

The Reverse card changes turn direction:

.. code-block:: text

    Clockwise (Normal):     1 → 2 → 3 → 4 → 1 → ...
    Counter-clockwise:      1 → 4 → 3 → 2 → 1 → ...

Implementation:

.. code-block:: python

    class MultiplayerUnoEnv:
        def __init__(self, num_players):
            self.direction = 1  # 1 = clockwise, -1 = counter-clockwise
            
        def _handle_reverse(self):
            self.direction *= -1
            
        def _next_player(self):
            return (self.current_player + self.direction) % self.num_players

Running Multiplayer Games
=========================

GUI Mode
--------

Launch the multiplayer GUI:

.. code-block:: bash

    python multiplayer_gui.py --players 4

Or from the main GUI:

1. Open ``uno_gui.py``
2. Click "Multiplayer (3-4)"
3. Select number of players

Programmatic Mode
-----------------

.. code-block:: python

    from src.multiplayer_env import MultiplayerUnoEnv
    from sb3_contrib import RecurrentPPO
    
    # Load models
    models = [
        RecurrentPPO.load("models/selfplay_champion.zip"),
        RecurrentPPO.load("models/best_recurrent_ppo.zip"),
        RecurrentPPO.load("models/optimal_recurrent_ppo.zip"),
        RecurrentPPO.load("models/sb3_recurrent_ppo.zip")
    ]
    
    # Create environment
    env = MultiplayerUnoEnv(num_players=4)
    
    # Game loop
    obs = env.reset()
    lstm_states = [None] * 4
    
    while True:
        player = env.current_player
        action, lstm_states[player] = models[player].predict(
            obs, state=lstm_states[player], deterministic=True
        )
        
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            print(f"Player {info['winner']} wins!")
            break

Battle Arena Multiplayer
------------------------

In the battle arena:

1. Click **3P** or **4P** button
2. Configure a model for each player slot
3. Run batch evaluation
4. Compare multi-player statistics

Strategy Considerations
=======================

3-Player Strategy
-----------------

- Watch both opponents' card counts
- Reverse is defensive (buys time)
- +2/+4 more impactful (one enemy at a time)
- Position matters (who's before/after you)

4-Player Strategy
-----------------

- Track all three opponents
- Reverse changes who gets targeted
- Alliances form naturally
- More variance in outcomes

Special Card Effects
--------------------

.. list-table::
   :header-rows: 1

   * - Card
     - 2-Player
     - 3-4 Player
   * - Reverse
     - Acts as Skip
     - Changes direction
   * - Skip
     - Opponent skipped
     - Next player skipped
   * - +2
     - Opponent draws 2
     - Next player draws 2
   * - +4
     - Opponent draws 4
     - Next player draws 4

Training Multiplayer Agents
===========================

Training in multiplayer requires special considerations:

.. code-block:: python

    from src.multiplayer_env import MultiplayerUnoEnv
    from sb3_contrib import RecurrentPPO
    
    # Create multiplayer environment
    env = MultiplayerUnoEnv(num_players=4)
    
    # Train agent
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=256,  # Longer episodes
        batch_size=64
    )
    
    model.learn(total_timesteps=2000000)
    model.save("models/multiplayer_champion.zip")

Evaluation
----------

Win rate interpretation changes:

- 2-player: 50% = equal skill
- 3-player: 33% = equal skill
- 4-player: 25% = equal skill

A 40% win rate in 4-player is **excellent** (1.6x expected)!

API Reference
=============

MultiplayerUnoEnv
-----------------

.. code-block:: python

    class MultiplayerUnoEnv(gym.Env):
        """
        UNO environment supporting 2-4 players.
        
        Parameters
        ----------
        num_players : int
            Number of players (2-4)
            
        Attributes
        ----------
        current_player : int
            Index of current player (0 to num_players-1)
        direction : int
            Turn direction (1=clockwise, -1=counter-clockwise)
        hands : List[List[Card]]
            Cards in each player's hand
        """
        
        def reset(self):
            """Reset game and return initial observation."""
            
        def step(self, action):
            """Execute action and return (obs, reward, done, truncated, info)."""
            
        def get_valid_actions(self):
            """Return list of valid action indices."""

Known Limitations
=================

Current multiplayer implementation notes:

1. **Human Player**: Always player 0 in GUI
2. **Training**: Agents trained in 2-player may underperform in 4-player
3. **Observation**: Can see opponent hand sizes but not specific cards
4. **Stacking**: +2/+4 stacking not implemented (official UNO rules)

Future Improvements
===================

Planned enhancements:

- [ ] Online multiplayer support
- [ ] Mixed human/AI games
- [ ] Tournament mode
- [ ] Elo rating system
- [ ] Replay saving/loading
