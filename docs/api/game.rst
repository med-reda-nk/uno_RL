===========
Game Module
===========

.. module:: src.game
   :synopsis: UNO game engine

The game module provides the core UNO game logic.

UnoGame Class
=============

.. class:: UnoGame(num_players=2)

   Main UNO game controller.
   
   :param num_players: Number of players (2-4)
   :type num_players: int

   .. attribute:: players
      :type: List[Player]
      
      List of player objects.

   .. attribute:: deck
      :type: Deck
      
      The draw pile.

   .. attribute:: discard_pile
      :type: List[Card]
      
      Cards that have been played.

   .. attribute:: current_player
      :type: int
      
      Index of the current player.

   .. attribute:: direction
      :type: int
      
      Turn direction (1 or -1).

   .. method:: deal_cards()
      
      Deal 7 cards to each player.

   .. method:: play_card(player_idx, card)
      
      Execute a card play.
      
      :param player_idx: Index of the player
      :param card: Card to play
      :raises InvalidPlayError: If the card cannot be played

   .. method:: draw_card(player_idx)
      
      Player draws a card from the deck.
      
      :param player_idx: Index of the player
      :returns: The drawn card
      :rtype: Card

   .. method:: get_winner()
      
      Check if there's a winner.
      
      :returns: Winning player index, or None if game ongoing
      :rtype: Optional[int]

   .. method:: get_valid_plays(player_idx)
      
      Get all valid cards a player can play.
      
      :param player_idx: Index of the player
      :returns: List of playable cards
      :rtype: List[Card]

   .. method:: is_valid_play(card)
      
      Check if a card can be played on the current discard.
      
      :param card: Card to check
      :returns: True if valid
      :rtype: bool

Example Usage
=============

.. code-block:: python

    from src.game import UnoGame
    
    # Create a 2-player game
    game = UnoGame(num_players=2)
    
    # Deal cards
    game.deal_cards()
    
    # Get valid plays for current player
    valid_cards = game.get_valid_plays(game.current_player)
    
    # Play a card
    if valid_cards:
        game.play_card(game.current_player, valid_cards[0])
    else:
        game.draw_card(game.current_player)
    
    # Check for winner
    winner = game.get_winner()
    if winner is not None:
        print(f"Player {winner} wins!")
