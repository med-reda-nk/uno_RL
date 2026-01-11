"""
UNO RL Project Setup
====================
Run this to set up the project with all dependencies.

Usage:
    python setup_project.py
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"  ✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} - FAILED")
        print(f"    Error: {e}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║            UNO RL Project Setup                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    print(f"  Python Version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("  ⚠️  Warning: Python 3.8+ is recommended")
    
    # Install requirements
    print("\n  📦 Installing Dependencies...")
    
    success = run_command(
        f'"{sys.executable}" -m pip install -r requirements.txt',
        "Installing Python packages"
    )
    
    if not success:
        print("\n  ❌ Failed to install some packages.")
        print("     Try running manually: pip install -r requirements.txt")
    
    # Verify installations
    print("\n\n  🔍 Verifying Installations...")
    
    packages = [
        ("pygame", "Pygame (GUI)"),
        ("stable_baselines3", "Stable Baselines3 (RL)"),
        ("sb3_contrib", "SB3 Contrib (RecurrentPPO)"),
        ("torch", "PyTorch (Deep Learning)"),
        ("numpy", "NumPy (Math)"),
        ("pandas", "Pandas (Data)"),
        ("matplotlib", "Matplotlib (Plots)"),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"    ✓ {name}")
        except ImportError:
            print(f"    ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    # Check models exist
    print("\n  🤖 Checking Trained Models...")
    
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        print(f"    ✓ Found {len(model_files)} trained models")
        
        # Check for best model
        if os.path.exists("models/best_recurrent_ppo_uno.zip"):
            print(f"    ✓ Best model (60% win rate) available")
    else:
        print(f"    ⚠️  Models directory not found")
    
    # Summary
    print("\n")
    print("  " + "="*56)
    
    if all_ok:
        print("""
    ✅ Setup Complete!
    
    Quick Start:
      python play.py           # Interactive menu
      python uno_gui.py        # Play against AI
      python model_battle_gui.py  # Watch AI battle
    
    For more info, see README.md
        """)
    else:
        print("""
    ⚠️  Setup completed with some issues.
    
    Try running: pip install -r requirements.txt
    
    Then run: python play.py
        """)


if __name__ == "__main__":
    main()
