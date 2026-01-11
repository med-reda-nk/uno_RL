"""
Deep Q-Network (DQN) Agent for UNO Game
A real reinforcement learning agent using neural networks and experience replay.
"""

import numpy as np
import random
from collections import deque
import pickle
import os

# Try to import PyTorch, fall back to numpy-based implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using NumPy-based neural network.")


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class NumpyNeuralNetwork:
    """Simple neural network implemented with NumPy for environments without PyTorch."""
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.lr = learning_rate
        self.layers = []
        
        # Initialize weights with He initialization
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.layers.append({'w': w, 'b': b})
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x, store_intermediates=False):
        """Forward pass through the network."""
        if store_intermediates:
            self.activations = [x]
            self.pre_activations = []
        
        current = x
        for i, layer in enumerate(self.layers[:-1]):
            z = current @ layer['w'] + layer['b']
            if store_intermediates:
                self.pre_activations.append(z)
            current = self.relu(z)
            if store_intermediates:
                self.activations.append(current)
        
        # Output layer (no activation)
        z = current @ self.layers[-1]['w'] + self.layers[-1]['b']
        if store_intermediates:
            self.pre_activations.append(z)
            self.activations.append(z)
        
        return z
    
    def backward(self, loss_grad):
        """Backward pass to compute gradients."""
        gradients = []
        delta = loss_grad
        
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                delta = delta * self.relu_derivative(self.pre_activations[i])
            
            grad_w = self.activations[i].T @ delta
            grad_b = np.sum(delta, axis=0, keepdims=True)
            gradients.insert(0, {'w': grad_w, 'b': grad_b})
            
            if i > 0:
                delta = delta @ self.layers[i]['w'].T
        
        return gradients
    
    def update(self, gradients):
        """Update weights using gradients."""
        for layer, grad in zip(self.layers, gradients):
            layer['w'] -= self.lr * grad['w']
            layer['b'] -= self.lr * grad['b']
    
    def copy_from(self, other):
        """Copy weights from another network."""
        for i, layer in enumerate(other.layers):
            self.layers[i]['w'] = layer['w'].copy()
            self.layers[i]['b'] = layer['b'].copy()


if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network using PyTorch."""
        
        def __init__(self, state_size, action_size, hidden_sizes=[128, 128, 64]):
            super(DQNNetwork, self).__init__()
            
            layers = []
            input_size = state_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                input_size = hidden_size
            
            layers.append(nn.Linear(input_size, action_size))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)


class DQNAgent:
    """
    Deep Q-Network Agent for playing UNO.
    
    This agent uses:
    - Neural network function approximation
    - Experience replay for stable learning
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration with decay
    """
    
    # Action mapping
    ACTION_MAP = {
        0: "RED", 1: "GRE", 2: "BLU", 3: "YEL",
        4: "SKI", 5: "REV", 6: "PL2", 7: "PL4", 8: "COL"
    }
    REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}
    
    def __init__(self, agent_info: dict):
        """
        Initialize the DQN Agent.
        
        Parameters:
        -----------
        agent_info : dict
            - epsilon: Initial exploration rate
            - epsilon_decay: Decay rate for epsilon
            - epsilon_min: Minimum epsilon value
            - gamma: Discount factor
            - learning_rate: Learning rate for optimizer
            - batch_size: Batch size for training
            - buffer_size: Size of replay buffer
            - target_update_freq: How often to update target network
            - hidden_sizes: List of hidden layer sizes
        """
        # Hyperparameters
        self.epsilon = agent_info.get("epsilon", 1.0)
        self.epsilon_decay = agent_info.get("epsilon_decay", 0.995)
        self.epsilon_min = agent_info.get("epsilon_min", 0.01)
        self.gamma = agent_info.get("gamma", 0.99)
        self.learning_rate = agent_info.get("learning_rate", 0.001)
        self.batch_size = agent_info.get("batch_size", 64)
        self.target_update_freq = agent_info.get("target_update_freq", 100)
        hidden_sizes = agent_info.get("hidden_sizes", [128, 128, 64])
        
        # State and action dimensions
        self.state_size = 17  # Based on the state representation in the game
        self.action_size = 9  # 9 possible actions
        
        # Initialize networks
        self.use_torch = TORCH_AVAILABLE
        
        if self.use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = DQNNetwork(self.state_size, self.action_size, hidden_sizes).to(self.device)
            self.target_net = DQNNetwork(self.state_size, self.action_size, hidden_sizes).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            self.policy_net = NumpyNeuralNetwork(self.state_size, hidden_sizes, self.action_size, self.learning_rate)
            self.target_net = NumpyNeuralNetwork(self.state_size, hidden_sizes, self.action_size, self.learning_rate)
            self.target_net.copy_from(self.policy_net)
        
        # Experience replay
        buffer_size = agent_info.get("buffer_size", 10000)
        self.memory = ReplayBuffer(buffer_size)
        
        # Training state
        self.prev_state = None
        self.prev_action = None
        self.steps = 0
        self.training_losses = []
        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def state_dict_to_vector(self, state_dict):
        """Convert state dictionary to a numpy vector."""
        # Color encoding (one-hot)
        color_map = {"RED": 0, "GRE": 1, "BLU": 2, "YEL": 3}
        color_vec = [0, 0, 0, 0]
        if state_dict.get("OPEN") in color_map:
            color_vec[color_map[state_dict["OPEN"]]] = 1
        
        # Card counts (normalized)
        cards = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL",
                 "RED#", "GRE#", "BLU#", "YEL#", "SKI#", "REV#", "PL2#"]
        
        card_values = []
        for card in cards[:-7]:  # First 9 cards (max value 2)
            card_values.append(state_dict.get(card, 0) / 2.0)
        for card in cards[-7:]:  # Last 7 cards (playable, max value 1-2)
            card_values.append(state_dict.get(card, 0) / 2.0)
        
        # Combine into state vector
        state_vector = np.array(color_vec + card_values, dtype=np.float32)
        
        # Ensure correct size
        if len(state_vector) < self.state_size:
            state_vector = np.pad(state_vector, (0, self.state_size - len(state_vector)))
        elif len(state_vector) > self.state_size:
            state_vector = state_vector[:self.state_size]
        
        return state_vector
    
    def get_valid_actions_mask(self, actions_dict):
        """Get a mask of valid actions."""
        mask = np.zeros(self.action_size)
        for action_name, available in actions_dict.items():
            if available != 0 and action_name in self.REVERSE_ACTION_MAP:
                action_idx = self.REVERSE_ACTION_MAP[action_name]
                mask[action_idx] = 1
        return mask
    
    def step(self, state_dict, actions_dict):
        """
        Choose an action using epsilon-greedy policy.
        
        Parameters:
        -----------
        state_dict : dict
            Current state representation
        actions_dict : dict
            Available actions with their validity
        
        Returns:
        --------
        str : Chosen action name
        """
        state_vector = self.state_dict_to_vector(state_dict)
        valid_mask = self.get_valid_actions_mask(actions_dict)
        
        # Get valid action indices
        valid_actions = np.where(valid_mask == 1)[0]
        
        if len(valid_actions) == 0:
            # No valid actions, return a random action from the action space
            return random.choice(list(self.ACTION_MAP.values()))
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_idx = random.choice(valid_actions)
        else:
            # Get Q-values from policy network
            if self.use_torch:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            else:
                q_values = self.policy_net.forward(state_vector.reshape(1, -1))[0]
            
            # Mask invalid actions with very negative values
            masked_q_values = q_values.copy()
            masked_q_values[valid_mask == 0] = -float('inf')
            
            action_idx = np.argmax(masked_q_values)
        
        action_name = self.ACTION_MAP[action_idx]
        
        # Store for learning
        self.prev_state = state_vector
        self.prev_action = action_idx
        
        return action_name
    
    def update(self, state_dict, action):
        """
        Store transition and perform learning update.
        
        Parameters:
        -----------
        state_dict : dict
            Current state after taking action
        action : str
            Action that was taken
        """
        if self.prev_state is None:
            return
        
        current_state = self.state_dict_to_vector(state_dict)
        action_idx = self.REVERSE_ACTION_MAP.get(action, 0)
        
        # Calculate reward
        reward = self.calculate_reward(state_dict, action)
        self.current_episode_reward += reward
        
        # Check if episode is done (player won or lost)
        done = self.is_terminal_state(state_dict)
        
        # Store transition
        self.memory.push(self.prev_state, self.prev_action, reward, current_state, done)
        
        # Learn from experience
        if len(self.memory) >= self.batch_size:
            loss = self.learn()
            self.training_losses.append(loss)
        
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Update previous state/action for next step
        self.prev_state = current_state
        self.prev_action = action_idx
    
    def calculate_reward(self, state_dict, action):
        """
        Calculate reward for a state-action pair.
        
        Reward structure:
        - Playing a card: +1
        - Playing special cards: +2
        - Playing wild cards: +1.5
        - Winning (no cards left): +100
        - Each card remaining: -0.1
        """
        reward = 0
        
        # Count remaining cards
        card_keys = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        total_cards = sum(state_dict.get(key, 0) for key in card_keys)
        
        # Winning condition
        if total_cards == 0:
            reward += 100
        else:
            # Penalty for remaining cards
            reward -= total_cards * 0.1
        
        # Reward for action type
        if action in ["SKI", "REV", "PL2"]:
            reward += 2  # Special cards
        elif action in ["PL4", "COL"]:
            reward += 1.5  # Wild cards
        else:
            reward += 1  # Normal cards
        
        return reward
    
    def is_terminal_state(self, state_dict):
        """Check if the current state is terminal (game over)."""
        card_keys = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        total_cards = sum(state_dict.get(key, 0) for key in card_keys)
        return total_cards == 0
    
    def learn(self):
        """Perform a learning step using experience replay."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        if self.use_torch:
            return self._learn_torch(states, actions, rewards, next_states, dones)
        else:
            return self._learn_numpy(states, actions, rewards, next_states, dones)
    
    def _learn_torch(self, states, actions, rewards, next_states, dones):
        """PyTorch learning implementation."""
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q values (using Double DQN)
        with torch.no_grad():
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Evaluate actions using target network
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def _learn_numpy(self, states, actions, rewards, next_states, dones):
        """NumPy learning implementation."""
        # Forward pass on current states
        current_q_all = self.policy_net.forward(states, store_intermediates=True)
        
        # Get Q-values for actions taken
        batch_indices = np.arange(len(actions))
        current_q = current_q_all[batch_indices, actions]
        
        # Get target Q-values
        next_q_policy = self.policy_net.forward(next_states)
        next_actions = np.argmax(next_q_policy, axis=1)
        next_q_target = self.target_net.forward(next_states)
        next_q = next_q_target[batch_indices, next_actions]
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute TD error
        td_error = current_q - target_q
        
        # Create gradient for output layer
        grad = np.zeros_like(current_q_all)
        grad[batch_indices, actions] = td_error / len(actions)
        
        # Backward pass and update
        gradients = self.policy_net.backward(grad)
        self.policy_net.update(gradients)
        
        return np.mean(td_error ** 2)
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        if self.use_torch:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net.copy_from(self.policy_net)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def end_episode(self, won: bool):
        """
        Called at the end of each episode.
        
        Parameters:
        -----------
        won : bool
            Whether the agent won the game
        """
        # Add final reward for winning/losing
        if won:
            self.current_episode_reward += 50
        else:
            self.current_episode_reward -= 20
        
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Reset previous state
        self.prev_state = None
        self.prev_action = None
    
    def save(self, filepath="models/dqn_agent.pkl"):
        """Save the agent to a file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        save_dict = {
            "epsilon": self.epsilon,
            "steps": self.steps,
            "training_losses": self.training_losses,
            "episode_rewards": self.episode_rewards,
        }
        
        if self.use_torch:
            save_dict["policy_net_state"] = self.policy_net.state_dict()
            save_dict["target_net_state"] = self.target_net.state_dict()
            save_dict["optimizer_state"] = self.optimizer.state_dict()
            torch.save(save_dict, filepath)
        else:
            save_dict["policy_net_layers"] = self.policy_net.layers
            save_dict["target_net_layers"] = self.target_net.layers
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f)
        
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath="models/dqn_agent.pkl"):
        """Load the agent from a file."""
        if not os.path.exists(filepath):
            print(f"No saved agent found at {filepath}")
            return False
        
        if self.use_torch:
            save_dict = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(save_dict["policy_net_state"])
            self.target_net.load_state_dict(save_dict["target_net_state"])
            self.optimizer.load_state_dict(save_dict["optimizer_state"])
        else:
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)
            self.policy_net.layers = save_dict["policy_net_layers"]
            self.target_net.layers = save_dict["target_net_layers"]
        
        self.epsilon = save_dict["epsilon"]
        self.steps = save_dict["steps"]
        self.training_losses = save_dict.get("training_losses", [])
        self.episode_rewards = save_dict.get("episode_rewards", [])
        
        print(f"Agent loaded from {filepath}")
        return True
    
    def get_training_stats(self):
        """Get training statistics."""
        stats = {
            "total_steps": self.steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.memory),
            "episodes": len(self.episode_rewards),
        }
        
        if self.training_losses:
            stats["avg_loss"] = np.mean(self.training_losses[-100:])
        
        if self.episode_rewards:
            stats["avg_reward"] = np.mean(self.episode_rewards[-100:])
            stats["max_reward"] = max(self.episode_rewards)
        
        return stats


class ImprovedQLearningAgent:
    """
    Improved Q-Learning Agent with better features:
    - Experience replay
    - Proper discount factor
    - Better reward shaping
    - State hashing for efficiency
    """
    
    ACTION_MAP = {
        0: "RED", 1: "GRE", 2: "BLU", 3: "YEL",
        4: "SKI", 5: "REV", 6: "PL2", 7: "PL4", 8: "COL"
    }
    REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}
    
    def __init__(self, agent_info: dict):
        self.epsilon = agent_info.get("epsilon", 0.3)
        self.epsilon_decay = agent_info.get("epsilon_decay", 0.9995)
        self.epsilon_min = agent_info.get("epsilon_min", 0.05)
        self.alpha = agent_info.get("step_size", 0.1)  # Learning rate
        self.gamma = agent_info.get("gamma", 0.95)  # Discount factor
        
        # Q-table using dictionary for efficiency
        self.q_table = {}
        self.visit_count = {}
        
        # Experience replay
        self.memory = ReplayBuffer(agent_info.get("buffer_size", 5000))
        self.batch_size = agent_info.get("batch_size", 32)
        
        # State tracking
        self.prev_state = None
        self.prev_action = None
        self.steps = 0
    
    def _state_to_key(self, state_dict):
        """Convert state dictionary to hashable key."""
        if isinstance(state_dict, dict):
            return tuple(state_dict.values())
        elif isinstance(state_dict, (tuple, list)):
            return tuple(state_dict)
        elif isinstance(state_dict, np.ndarray):
            return tuple(state_dict.tolist())
        return state_dict
    
    def _get_q_value(self, state_key, action):
        """Get Q-value for state-action pair."""
        # Convert to tuple if needed
        if isinstance(state_key, np.ndarray):
            state_key = tuple(state_key.tolist())
        elif not isinstance(state_key, tuple):
            state_key = tuple(state_key) if hasattr(state_key, '__iter__') else state_key
        return self.q_table.get((state_key, action), 0.0)
    
    def _set_q_value(self, state_key, action, value):
        """Set Q-value for state-action pair."""
        # Convert to tuple if needed
        if isinstance(state_key, np.ndarray):
            state_key = tuple(state_key.tolist())
        elif not isinstance(state_key, tuple):
            state_key = tuple(state_key) if hasattr(state_key, '__iter__') else state_key
        self.q_table[(state_key, action)] = value
        self.visit_count[(state_key, action)] = self.visit_count.get((state_key, action), 0) + 1
    
    def step(self, state_dict, actions_dict):
        """Choose action using epsilon-greedy."""
        state_key = self._state_to_key(state_dict)
        valid_actions = [k for k, v in actions_dict.items() if v != 0]
        
        if not valid_actions:
            return random.choice(list(self.ACTION_MAP.values()))
        
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            # Get Q-values for valid actions
            q_values = {a: self._get_q_value(state_key, a) for a in valid_actions}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        
        self.prev_state = state_key
        self.prev_action = action
        
        return action
    
    def update(self, state_dict, action):
        """Update Q-values using Q-learning update rule."""
        if self.prev_state is None:
            return
        
        current_state = self._state_to_key(state_dict)
        
        # Calculate reward
        reward = self._calculate_reward(state_dict, action)
        done = self._is_terminal(state_dict)
        
        # Store experience
        self.memory.push(self.prev_state, self.prev_action, reward, current_state, done)
        
        # Q-learning update
        prev_q = self._get_q_value(self.prev_state, self.prev_action)
        
        if done:
            target = reward
        else:
            # Get max Q-value for next state
            next_q_values = [self._get_q_value(current_state, a) for a in self.ACTION_MAP.values()]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.gamma * max_next_q
        
        # Update Q-value
        new_q = prev_q + self.alpha * (target - prev_q)
        self._set_q_value(self.prev_state, self.prev_action, new_q)
        
        self.steps += 1
        
        # Experience replay update
        if len(self.memory) >= self.batch_size:
            self._replay()
    
    def _replay(self):
        """Learn from past experiences."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            
            prev_q = self._get_q_value(state, action)
            
            if done:
                target = reward
            else:
                next_q_values = [self._get_q_value(next_state, a) for a in self.ACTION_MAP.values()]
                max_next_q = max(next_q_values) if next_q_values else 0
                target = reward + self.gamma * max_next_q
            
            new_q = prev_q + self.alpha * (target - prev_q)
            self._set_q_value(state, action, new_q)
    
    def _calculate_reward(self, state_dict, action):
        """Calculate reward for the transition."""
        reward = 0
        card_keys = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        total_cards = sum(state_dict.get(key, 0) for key in card_keys)
        
        if total_cards == 0:
            reward += 100
        else:
            reward -= total_cards * 0.1
        
        if action in ["SKI", "REV", "PL2"]:
            reward += 2
        elif action in ["PL4", "COL"]:
            reward += 1.5
        else:
            reward += 1
        
        return reward
    
    def _is_terminal(self, state_dict):
        """Check if state is terminal."""
        card_keys = ["RED", "GRE", "BLU", "YEL", "SKI", "REV", "PL2", "PL4", "COL"]
        return sum(state_dict.get(key, 0) for key in card_keys) == 0
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def end_episode(self, won: bool):
        """End of episode handling."""
        self.decay_epsilon()
        self.prev_state = None
        self.prev_action = None
    
    def save(self, filepath="models/qlearning_agent.pkl"):
        """Save the agent."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        save_dict = {
            "q_table": self.q_table,
            "visit_count": self.visit_count,
            "epsilon": self.epsilon,
            "steps": self.steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath="models/qlearning_agent.pkl"):
        """Load the agent."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        self.q_table = save_dict["q_table"]
        self.visit_count = save_dict["visit_count"]
        self.epsilon = save_dict["epsilon"]
        self.steps = save_dict["steps"]
        print(f"Agent loaded from {filepath}")
        return True
    
    def get_training_stats(self):
        """Get training statistics."""
        return {
            "total_steps": self.steps,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "buffer_size": len(self.memory)
        }
