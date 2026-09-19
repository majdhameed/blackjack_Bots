# Card and shoe primitives. A shoe contains one or more shuffled decks and
# removes cards as they are dealt.
import random

RANKS = [2, 3, 4, 5, 6, 7 , 8, 9, 10, 'J', 'Q', 'K', 'A']
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

class Card:
    def __init__(self, rank, suit):
        if rank not in RANKS:
            raise ValueError("This is not a valid card rank!")
        if suit not in SUITS:
            raise ValueError("This is not a valid card suit")     
        self.rank = rank
        self.suit = suit

    def get_value(self):
        # Aces begin at 11; Hand reduces them to 1 when needed to avoid a bust.
        if self.rank == 'A':
            return 11
        if self.rank in ['J', 'Q', 'K']:
            return 10
        return self.rank

    def __str__(self):
        return (f"{self.rank} of {self.suit}")

class Shoe:
    def __init__(self, decks):
        if decks < 1:
            raise ValueError("A shoe must have at least 1 deck")
        self.decks = decks
        self.cards = []

        # Build every rank/suit combination once for each requested deck.
        i = 0

        while i < decks:
            for suit in SUITS:
                for rank in RANKS:
                    card = Card(rank, suit)
                    self.cards.append(card)
            i += 1
        
        random.shuffle(self.cards)

    def deal_card(self):
        if len(self.cards) == 0:
            raise ValueError("There are no cards to deal!")

        # Popping avoids repeatedly shifting the rest of the list.
        return self.cards.pop()

    def cards_remaining(self):
        return len(self.cards)
        



