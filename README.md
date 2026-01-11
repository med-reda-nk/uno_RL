# UNO Card Game with Reinforcement Learning

A complete UNO card game implementation with trained RL agents using Stable Baselines3.

- models link : [models](https://drive.google.com/drive/folders/15scFJZZgl2_fWGNbJ91UexQvjNqjRhEy?usp=drive_link)

- report link : [Docs](https://uno-rl.readthedocs.io/en/latest/)
- 
## Quick Start

```bash
# Play against the AI
python uno_gui.py

# Watch AI models battle each other
python model_battle_gui.py

# List available commands
python run.py --mode list
```

## Trained Models (Ranked by Win Rate)

| Rank | Model | Win Rate | Type |
|------|-------|----------|------|
| 🏆 **1st** | **Self-Play Champion** | **70%+** | LSTM-PPO (Self-Play) |
| 2nd | Best Recurrent PPO | 60% | LSTM-PPO |
| 3rd | Optimal Recurrent PPO | 59% | LSTM-PPO |
| 4th | SB3 Recurrent PPO | 57% | LSTM-PPO |
| 5th | Enhanced RPPO | 56% | LSTM-PPO |
| 6th | Best PPO | 53% | PPO |
| 7th | PPO | 52% | PPO |
| 8th | DQN | 48% | DQN |
| 9th | A2C | 45% | A2C |

### Best Model: Self-Play Champion 🏆

The **Self-Play Champion** (`models/selfplay_champion.zip`) is our best performing model, achieving 70%+ win rate against random opponents. It was trained using self-play with curriculum learning.

```python
# Load and use the Self-Play Champion
from sb3_contrib import RecurrentPPO
model = RecurrentPPO.load("models/selfplay_champion.zip")
```

## Project Structure

```
uno-card-game-rl/
│
├── uno_gui.py              # Main game GUI - play against AI
├── multiplayer_gui.py      # NEW: 3-4 player multiplayer mode
├── model_battle_gui.py     # Model vs Model battle arena
├── run.py                  # CLI runner (eval, list, gui, battle)
├── compare_models.py       # Evaluate and compare all models
├── config.py               # Configuration and hyperparameters
├── play.py                 # Interactive launcher menu
├── README.md
│
├── training/               # All training scripts
│   ├── train_selfplay.py          # NEW: Self-play & curriculum learning
│   ├── train_best_recurrent_ppo.py
│   ├── train_optimal_recurrent_ppo.py
│   ├── train_enhanced_rppo.py
│   ├── train_recurrent_ppo.py
│   ├── train_sb3.py
│   ├── train_best_ppo.py
│   └── train_rl.py
│
├── src/                    # Core game engine
│   ├── multiplayer_env.py # NEW: 2-4 player environment
│   ├── sb3_agent.py       # SB3 Gym environment
│   ├── game.py            # Game logic
│   ├── cards.py           # Card/Deck classes
│   ├── players.py         # Player class
│   ├── turn.py            # Turn management
│   ├── dqn_agent.py       # Custom DQN implementation
│   ├── agents.py          # Legacy Q-learning/Monte-Carlo
│   ├── state_action_reward.py
│   └── utils.py
│
├── models/                 # Saved trained models
│   ├── selfplay_champion.zip          # 🏆 BEST (70%+) - Self-Play Champion
│   ├── best_recurrent_ppo_uno.zip     # 2nd best (60%)
│   ├── optimal_recurrent_ppo.zip      # 3rd best (59%)
│   ├── sb3_recurrentppo_uno.zip       # 4th best (57%)
│   ├── enhanced_rppo.zip
│   ├── sb3_ppo_uno.zip
│   ├── sb3_dqn_uno.zip
│   └── sb3_a2c_uno.zip
│
├── tests/                  # Unit tests
├── comparison_results/     # Model evaluation results (CSV/PNG)
├── notebooks/              # Analysis Jupyter notebooks
├── assets/                 # Training data and curves
└── logs/                   # TensorBoard training logs
```

## Usage

### Play the Game
```bash
python uno_gui.py
```
- **Model Selector**: Choose any AI opponent from the dropdown menu
- Modern UI with animations
- **Watch Mode**: See AI vs AI gameplay
- **Multiplayer Button**: Quick launch 3-4 player mode

### Model Battle Arena (Updated!)
```bash
python model_battle_gui.py
```
- **2-4 Player Battles**: Click 2P/3P/4P buttons to set player count
- **Multiple Model Selectors**: Assign different models to each player
- Run batch battles (100 games)
- Save results to CSV with per-player statistics

### Multiplayer Mode
```bash
# Play against 3 AI opponents (4 players total)
python multiplayer_gui.py --players 4

# Play against 2 AI opponents (3 players total)
python multiplayer_gui.py --players 3
```
- 3 or 4 player modes
- Multiple AI opponents
- Visual hand display for all players
- Turn indicator and direction arrows

### Command Line Interface
```bash
# List all models
python run.py --mode list

# Evaluate a specific model
python run.py --mode eval --model best_recurrent_ppo --games 100

# Launch GUIs
python run.py --mode gui
python run.py --mode battle

# Run legacy Q-learning
python run.py --mode legacy
```

### Compare All Models
```bash
python compare_models.py
```

## Training

### Self-Play Training (NEW! - Target: 70%+ Win Rate)
```bash
# Self-play with curriculum learning (recommended)
python training/train_selfplay.py --mode selfplay --timesteps 1000000

# Self-play for multiplayer (3-4 players)
python training/train_selfplay.py --mode selfplay --players 4 --timesteps 1000000

# Population-based training
python training/train_selfplay.py --mode population --players 4
```

### Train Best Recurrent PPO
```bash
python training/train_best_recurrent_ppo.py
```

### Train Other Models
```bash
# Recurrent PPO variants
python training/train_optimal_recurrent_ppo.py
python training/train_enhanced_rppo.py
python training/train_recurrent_ppo.py --timesteps 500000

# Standard SB3 models
python training/train_sb3.py --algorithm ppo --timesteps 200000
python training/train_sb3.py --algorithm dqn --timesteps 200000
python training/train_sb3.py --algorithm a2c --timesteps 200000

# Custom implementations
python training/train_rl.py --agent dqn --episodes 1000
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/uno-card-game-rl.git
cd uno-card-game-rl

# Create virtual environment
python -m venv .venv1
.venv1\Scripts\activate  # Windows
source .venv1/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
python setup_project.py
```

### Requirements
- Python 3.10+
- pygame
- stable-baselines3
- sb3-contrib (for RecurrentPPO)
- numpy, pandas, matplotlib

## Testing

Run unit tests to verify everything works:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_cards.py -v
python -m pytest tests/test_environment.py -v
```

## Key Features

- **Multiplayer Support** - Play with 2-4 players (NEW!)
- **Self-Play Training** - Train against copies of itself for better performance (NEW!)
- **Curriculum Learning** - Gradually increase opponent difficulty (NEW!)
- **Modern GUI** - Clean interface with animations
- **Multiple RL Algorithms** - PPO, DQN, A2C, Recurrent PPO
- **LSTM Memory** - Recurrent models remember game history
- **Model Comparison** - Battle arena for model vs model
- **Configurable** - Easy hyperparameter tuning via config.py
- **Well-Tested** - Unit tests for core game logic
- **Type Hints** - Modern Python with full type annotations

## Results

The Recurrent PPO models with LSTM memory significantly outperform standard models:
- LSTM allows tracking played cards and opponent patterns
- 60% win rate vs random opponent (vs 50% baseline)
- Enhanced reward shaping improved convergence

## Future Improvements

| Level | Improvement | Description |
|-------|-------------|-------------|
| 🟢 Easy | **Add logging** | Replace `print()` with `logging` module for better debugging |
| 🟢 Easy | **Add `pyproject.toml`** | Modern Python packaging standard |
| 🟡 Medium | **Add CI/CD** | GitHub Actions to run tests automatically |
| 🟡 Medium | **Add multiplayer** | Allow 3-4 players instead of just 2 |
| 🟡 Medium | **Improve win rate** | Train with self-play or curriculum learning (target: 70%+) |
| 🔴 Advanced | **Web version** | Convert Pygame to web app (Flask + JS or Streamlit) |
| 🔴 Advanced | **Human vs Human online** | Add networking for online play |

## Credits

Original project by [Bernhard Pfann](https://www.linkedin.com/in/bernhard-pfann/)  
Article: [Tackling UNO with Reinforcement Learning](https://towardsdatascience.com/tackling-uno-card-game-with-reinforcement-learning-fad2fc19355c)
