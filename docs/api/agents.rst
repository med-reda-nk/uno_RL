=============
Agents Module
=============

.. module:: src.agents
   :synopsis: RL agent implementations

The agents module provides various RL agent implementations for playing UNO.

Base Agent Class
================

.. class:: RLAgent

   Abstract base class for all RL agents.

   .. method:: select_action(obs, valid_actions)
      :abstractmethod:
      
      Select an action given observation and valid actions.
      
      :param obs: Current observation
      :type obs: np.ndarray
      :param valid_actions: List of valid action indices
      :type valid_actions: List[int]
      :returns: Selected action index
      :rtype: int

   .. method:: reset()
      
      Reset agent state (for stateful agents like LSTM).

RandomAgent Class
=================

.. class:: RandomAgent

   Agent that selects random valid actions.
   
   Useful as a baseline for evaluation.

   .. method:: select_action(obs, valid_actions)
      
      Select a random valid action.
      
      :param obs: Current observation (ignored)
      :param valid_actions: List of valid action indices
      :returns: Random action from valid_actions
      :rtype: int

SB3Agent Class
==============

.. class:: SB3Agent(model_path, deterministic=True)

   Wrapper for Stable-Baselines3 models.
   
   :param model_path: Path to saved model file (.zip)
   :type model_path: str
   :param deterministic: Whether to use deterministic actions
   :type deterministic: bool

   .. attribute:: model
      :type: BaseAlgorithm
      
      The loaded SB3 model.

   .. attribute:: lstm_state
      :type: Optional[Tuple]
      
      Hidden state for recurrent models.

   .. method:: select_action(obs, valid_actions)
      
      Select action using the trained model.
      
      :param obs: Current observation
      :param valid_actions: Valid actions (used for masking if needed)
      :returns: Action from model prediction
      :rtype: int

   .. method:: reset()
      
      Reset LSTM hidden state.

   .. method:: load(model_path)
      :classmethod:
      
      Load a model from file.
      
      :param model_path: Path to model file
      :returns: New SB3Agent instance
      :rtype: SB3Agent

DQNAgent Class
==============

.. class:: DQNAgent(model_path=None, state_size=17, action_size=9)

   Deep Q-Network agent.
   
   :param model_path: Path to saved model (optional)
   :param state_size: Observation dimension
   :param action_size: Number of actions

   .. attribute:: epsilon
      :type: float
      
      Exploration rate for epsilon-greedy.

   .. attribute:: memory
      :type: deque
      
      Experience replay buffer.

   .. method:: select_action(obs, valid_actions)
      
      Select action using epsilon-greedy policy.

   .. method:: train(batch_size=64)
      
      Train on a batch from replay buffer.

   .. method:: save(path)
      
      Save model weights.

   .. method:: load(path)
      
      Load model weights.

Recurrent Agents
================

For LSTM-based agents, the hidden state persists across timesteps:

.. code-block:: python

    class RecurrentAgent(SB3Agent):
        """Agent with LSTM memory."""
        
        def __init__(self, model_path):
            super().__init__(model_path)
            self.lstm_state = None
            
        def select_action(self, obs, valid_actions):
            action, self.lstm_state = self.model.predict(
                obs, 
                state=self.lstm_state,
                deterministic=True
            )
            return action
            
        def reset(self):
            self.lstm_state = None

Example Usage
=============

Random Agent
------------

.. code-block:: python

    from src.agents import RandomAgent
    from src.state_action_reward import UnoEnv
    
    env = UnoEnv()
    agent = RandomAgent()
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        valid = env.get_valid_actions()
        action = agent.select_action(obs, valid)
        obs, reward, done, _, _ = env.step(action)

SB3 Agent
---------

.. code-block:: python

    from src.agents import SB3Agent
    from src.state_action_reward import UnoEnv
    
    env = UnoEnv()
    agent = SB3Agent("models/selfplay_champion.zip")
    
    obs, _ = env.reset()
    agent.reset()  # Clear LSTM state
    done = False
    
    while not done:
        valid = env.get_valid_actions()
        action = agent.select_action(obs, valid)
        obs, reward, done, _, _ = env.step(action)
    
    print(f"Result: {'Win' if reward > 0 else 'Loss'}")

Comparing Agents
----------------

.. code-block:: python

    from src.agents import SB3Agent, RandomAgent
    from src.state_action_reward import UnoEnv
    
    def evaluate(agent1, agent2, env, num_games=100):
        wins = 0
        for _ in range(num_games):
            obs, _ = env.reset()
            agent1.reset()
            done = False
            
            while not done:
                action = agent1.select_action(obs, env.get_valid_actions())
                obs, reward, done, _, _ = env.step(action)
            
            if reward > 0:
                wins += 1
        
        return wins / num_games
    
    env = UnoEnv()
    champion = SB3Agent("models/selfplay_champion.zip")
    random = RandomAgent()
    
    win_rate = evaluate(champion, random, env)
    print(f"Champion win rate: {win_rate:.1%}")

Agent Comparison Table
======================

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Agent
     - Type
     - Win Rate
     - Notes
   * - SB3Agent (Self-Play)
     - LSTM
     - 70%+
     - Best performance
   * - SB3Agent (RecPPO)
     - LSTM
     - 60%
     - Good baseline
   * - SB3Agent (PPO)
     - MLP
     - 53%
     - No memory
   * - DQNAgent
     - MLP
     - 48%
     - Value-based
   * - RandomAgent
     - None
     - 25%
     - Baseline
