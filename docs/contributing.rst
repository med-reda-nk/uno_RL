============
Contributing
============

Thank you for your interest in contributing to UNO Card Game RL!

Getting Started
===============

1. Fork the Repository
----------------------

Fork the repository on GitHub and clone your fork:

.. code-block:: bash

    git clone https://github.com/YOUR_USERNAME/uno-card-game-rl.git
    cd uno-card-game-rl

2. Set Up Development Environment
---------------------------------

.. code-block:: bash

    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    
    # Install dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

3. Create a Branch
------------------

.. code-block:: bash

    git checkout -b feature/your-feature-name

Development Guidelines
======================

Code Style
----------

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints where possible
- Docstrings for all public functions/classes

.. code-block:: python

    def calculate_reward(done: bool, winner: int) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            done: Whether the game is finished
            winner: Index of winning player (-1 if ongoing)
            
        Returns:
            Reward value: 1.0 for win, -1.0 for loss, 0.0 otherwise
        """
        if not done:
            return 0.0
        return 1.0 if winner == 0 else -1.0

Testing
-------

Write tests for new features:

.. code-block:: bash

    # Run all tests
    pytest tests/ -v
    
    # Run specific test file
    pytest tests/test_game.py -v
    
    # Run with coverage
    pytest tests/ --cov=src --cov-report=html

Documentation
-------------

Update documentation for any changes:

.. code-block:: bash

    # Build docs locally
    cd docs
    make html
    
    # View in browser
    open _build/html/index.html

Pull Request Process
====================

1. Ensure Tests Pass
--------------------

.. code-block:: bash

    pytest tests/ -v

2. Update Documentation
-----------------------

- Update docstrings
- Update relevant .rst files
- Add to CHANGELOG if significant

3. Submit PR
------------

- Provide clear description of changes
- Reference any related issues
- Include screenshots for GUI changes

4. Code Review
--------------

- Address reviewer feedback
- Keep commits clean and focused

Types of Contributions
======================

Bug Fixes
---------

- Check existing issues first
- Include test case that reproduces bug
- Explain the fix in PR description

New Features
------------

- Discuss in issue first for major features
- Include tests and documentation
- Follow existing code patterns

Algorithm Improvements
----------------------

- Include benchmark comparisons
- Document hyperparameters used
- Provide training logs

Documentation
-------------

- Fix typos and clarify explanations
- Add examples and tutorials
- Improve API documentation

Areas for Contribution
======================

We especially welcome contributions in:

1. **New RL Algorithms**: Implement and benchmark new algorithms
2. **Hyperparameter Tuning**: Find better configurations
3. **GUI Improvements**: Better graphics, animations
4. **Testing**: Increase test coverage
5. **Documentation**: Tutorials, examples
6. **Performance**: Optimization, profiling

Code of Conduct
===============

Please be respectful and constructive in all interactions.

- Be welcoming to newcomers
- Provide helpful feedback
- Focus on the code, not the person
- Assume good intentions

Questions?
==========

- Open an issue for bugs or features
- Use discussions for questions
- Check existing issues before posting

Thank you for contributing! 🎮
