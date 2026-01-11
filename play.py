#!/usr/bin/env python
"""
UNO RL Quick Start Launcher
===========================
Run this script to launch the game menu.

Usage:
    python play.py          # Interactive menu
    python play.py --game   # Start game immediately
    python play.py --battle # Start model battle immediately
"""

import sys
import os

def print_banner():
    """Print the UNO ASCII banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ██╗   ██╗███╗   ██╗ ██████╗     ██████╗ ██╗          ║
    ║     ██║   ██║████╗  ██║██╔═══██╗    ██╔══██╗██║          ║
    ║     ██║   ██║██╔██╗ ██║██║   ██║    ██████╔╝██║          ║
    ║     ██║   ██║██║╚██╗██║██║   ██║    ██╔══██╗██║          ║
    ║     ╚██████╔╝██║ ╚████║╚██████╔╝    ██║  ██║███████╗     ║
    ║      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝     ╚═╝  ╚═╝╚══════╝     ║
    ║                                                          ║
    ║          Reinforcement Learning Card Game                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print the main menu."""
    print("""
    ┌─────────────────────────────────────┐
    │           MAIN MENU                 │
    ├─────────────────────────────────────┤
    │  [1] Play Against AI                │
    │  [2] Watch AI Battle                │
    │  [3] List Models                    │
    │  [4] Run Analysis Notebook          │
    │  [5] Train New Model                │
    │  [Q] Quit                           │
    └─────────────────────────────────────┘
    """)

def launch_game():
    """Launch the main UNO game GUI."""
    print("\n  🎮 Launching UNO Game...")
    os.system(f'"{sys.executable}" uno_gui.py')

def launch_battle():
    """Launch the model battle arena."""
    print("\n  ⚔️ Launching Model Battle Arena...")
    os.system(f'"{sys.executable}" model_battle_gui.py')

def list_models():
    """List available models."""
    print("\n  📋 Available Models:\n")
    os.system(f'"{sys.executable}" run.py --mode list')
    input("\n  Press Enter to continue...")

def open_notebook():
    """Open the analysis notebook."""
    print("\n  📊 Opening Analysis Notebook...")
    os.system('jupyter notebook notebooks/model_analysis.ipynb')

def train_model():
    """Show training options."""
    print("""
    ┌─────────────────────────────────────┐
    │        TRAINING OPTIONS             │
    ├─────────────────────────────────────┤
    │  [1] Train Self-Play Champion       │
    │  [2] Train Best PPO                 │
    │  [3] Train Recurrent PPO            │
    │  [4] Train Optimal Recurrent PPO    │
    │  [B] Back to Main Menu              │
    └─────────────────────────────────────┘
    """)
    
    choice = input("  Enter choice: ").strip().upper()
    
    if choice == "1":
        os.system(f'"{sys.executable}" training/train_selfplay.py')
    elif choice == "2":
        os.system(f'"{sys.executable}" training/train_best_ppo.py')
    elif choice == "3":
        os.system(f'"{sys.executable}" training/train_recurrent_ppo.py')
    elif choice == "4":
        os.system(f'"{sys.executable}" training/train_optimal_recurrent_ppo.py')

def main():
    """Main entry point."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--game', '-g']:
            launch_game()
            return
        elif arg in ['--battle', '-b']:
            launch_battle()
            return
        elif arg in ['--help', '-h']:
            print(__doc__)
            return
    
    # Interactive menu
    print_banner()
    
    while True:
        print_menu()
        choice = input("  Enter choice: ").strip().upper()
        
        if choice == "1":
            launch_game()
        elif choice == "2":
            launch_battle()
        elif choice == "3":
            list_models()
        elif choice == "4":
            open_notebook()
        elif choice == "5":
            train_model()
        elif choice in ["Q", "QUIT", "EXIT"]:
            print("\n  Thanks for playing UNO! 👋\n")
            break
        else:
            print("\n  ❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
