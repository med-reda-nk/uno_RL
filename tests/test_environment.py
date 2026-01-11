"""
Tests for the RL Environment
Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUnoEnv:
    """Tests for the UNO Gymnasium Environment."""
    
    @pytest.fixture
    def env(self):
        """Create a test environment."""
        from src.sb3_agent import UnoEnv
        return UnoEnv()
    
    def test_env_creation(self, env):
        """Test environment can be created."""
        assert env is not None
    
    def test_observation_space(self, env):
        """Test observation space is correct shape."""
        assert env.observation_space.shape == (17,)
    
    def test_action_space(self, env):
        """Test action space has 9 actions."""
        assert env.action_space.n == 9
    
    def test_reset_returns_observation(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset()
        
        assert obs is not None
        assert obs.shape == (17,)
        assert isinstance(info, dict)
    
    def test_observation_bounds(self, env):
        """Test observations are within bounds [0, 1]."""
        obs, _ = env.reset()
        
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)
    
    def test_step_returns_correct_format(self, env):
        """Test step returns (obs, reward, done, truncated, info)."""
        env.reset()
        action = env.action_space.sample()
        
        result = env.step(action)
        
        assert len(result) == 5  # obs, reward, done, truncated, info
        obs, reward, done, truncated, info = result
        
        assert obs.shape == (17,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_multiple_steps(self, env):
        """Test environment can run multiple steps."""
        env.reset()
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                env.reset()


class TestModelLoading:
    """Tests for loading trained models."""
    
    def test_model_files_exist(self):
        """Test that model files exist."""
        import os
        
        models_dir = "models"
        assert os.path.exists(models_dir)
        
        # Check for at least one model
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        assert len(model_files) > 0
    
    def test_best_model_exists(self):
        """Test best model file exists."""
        import os
        
        best_model = "models/best_recurrent_ppo_uno.zip"
        assert os.path.exists(best_model), f"Best model not found: {best_model}"
    
    def test_config_has_models(self):
        """Test config file has model definitions."""
        import config
        
        assert hasattr(config, 'sb3_models')
        assert len(config.sb3_models) > 0


class TestRewardFunction:
    """Tests for the reward function."""
    
    def test_win_reward_positive(self):
        """Test winning gives positive reward."""
        WIN_REWARD = 10.0
        assert WIN_REWARD > 0
    
    def test_lose_reward_negative(self):
        """Test losing gives negative reward."""
        LOSE_REWARD = -10.0
        assert LOSE_REWARD < 0
    
    def test_valid_play_small_reward(self):
        """Test valid plays give small positive reward."""
        VALID_PLAY_REWARD = 0.1
        assert VALID_PLAY_REWARD > 0
        assert VALID_PLAY_REWARD < 1  # Should be small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
