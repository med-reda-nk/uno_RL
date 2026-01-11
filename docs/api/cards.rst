============
Cards Module
============

.. module:: src.cards
   :synopsis: Card and deck definitions

The cards module defines UNO cards and the deck.

Card Class
==========

.. class:: Card(color, value)

   Represents a single UNO card.
   
   :param color: Card color ('red', 'green', 'blue', 'yellow', 'wild')
   :type color: str
   :param value: Card value ('0'-'9', 'skip', 'reverse', 'draw2', 'wild', 'draw4')
   :type value: str

   .. attribute:: color
      :type: str
      
      The card's color.

   .. attribute:: value
      :type: str
      
      The card's value/type.

   .. method:: is_number()
      
      Check if this is a number card (0-9).
      
      :returns: True if number card
      :rtype: bool

   .. method:: is_special()
      
      Check if this is a special action card.
      
      :returns: True if skip, reverse, or draw2
      :rtype: bool

   .. method:: is_wild()
      
      Check if this is a wild card.
      
      :returns: True if wild or draw4
      :rtype: bool

   .. method:: matches(other)
      
      Check if this card can be played on another.
      
      :param other: Card to compare with
      :type other: Card
      :returns: True if playable
      :rtype: bool

Deck Class
==========

.. class:: Deck()

   Represents a standard 108-card UNO deck.
   
   The deck contains:
   
   - 76 number cards (0-9 in each color, 0 appears once, 1-9 twice)
   - 24 action cards (2 each of Skip, Reverse, Draw2 per color)
   - 8 wild cards (4 Wild, 4 Wild Draw4)

   .. method:: shuffle()
      
      Shuffle the deck.

   .. method:: draw()
      
      Draw the top card from the deck.
      
      :returns: The drawn card
      :rtype: Card
      :raises EmptyDeckError: If deck is empty

   .. method:: add_cards(cards)
      
      Add cards to the bottom of the deck.
      
      :param cards: Cards to add
      :type cards: List[Card]

   .. method:: is_empty()
      
      Check if deck is empty.
      
      :returns: True if no cards remain
      :rtype: bool

   .. attribute:: cards
      :type: List[Card]
      
      The remaining cards in the deck.

Card Constants
==============

.. data:: COLORS

   Valid card colors: ``['red', 'green', 'blue', 'yellow']``

.. data:: NUMBERS

   Number card values: ``['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']``

.. data:: ACTIONS

   Action card types: ``['skip', 'reverse', 'draw2']``

.. data:: WILDS

   Wild card types: ``['wild', 'draw4']``

Example Usage
=============

.. code-block:: python

    from src.cards import Card, Deck, COLORS
    
    # Create individual cards
    red_seven = Card('red', '7')
    blue_skip = Card('blue', 'skip')
    wild_card = Card('wild', 'wild')
    
    # Check card properties
    print(red_seven.is_number())   # True
    print(blue_skip.is_special())  # True
    print(wild_card.is_wild())     # True
    
    # Check if cards match
    red_five = Card('red', '5')
    print(red_seven.matches(red_five))  # True (same color)
    
    green_seven = Card('green', '7')
    print(red_seven.matches(green_seven))  # True (same value)
    
    # Create and use deck
    deck = Deck()
    deck.shuffle()
    
    # Draw cards
    hand = [deck.draw() for _ in range(7)]
    print(f"Hand: {hand}")
