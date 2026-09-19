import math

from blackjack.actions import Action
from blackjack.cards import Shoe
from blackjack.player import Player
from tournament.table_round import TableRound
from tournament.observation import BettingObservation, ActionObservation


class Tournament:
    def __init__(self, players, bots, number_of_rounds, decks, minimum_bet, hit_soft_17, max_hands):

        if not isinstance(players, list) or not isinstance(bots, list):
            raise TypeError("Players and bots must be provided as a list")
        if len(bots) != len(players):
            raise ValueError("There are unequal numbers of players and bots")
        if len(players) < 2:
            raise ValueError("There must be atleast 2 players")
        if number_of_rounds <= 0:
            raise ValueError("The number of rounds must be a positive integer")
        for player in players:
            if not isinstance(player, Player):
                raise TypeError("player must be of type Player") 
        for bot in bots:
            if not callable(getattr(bot, "choose_bet", None)):
                raise TypeError(
                    "Each bot must have a choose_bet method"
                )

            if not callable(getattr(bot, "choose_action", None)):
                raise TypeError(
                    "Each bot must have a choose_action method"
                )        
        
        self.players = players
        self.bots = bots
        self.number_of_rounds = number_of_rounds
        self.minimum_bet = minimum_bet
        self.hit_soft_17 = hit_soft_17
        self.max_hands = max_hands

        self.shoe = Shoe(decks)
        self.current_round_number = 0
        self.starting_player_index = 0
        self.round_history = []
        self.is_over = False
        self.current_table_round = None
        self.active_player_indices = []




    def get_active_player_indices(self):
        indices = []
        for player_index, player in enumerate(self.players):
            if player.bankroll > 0:
                indices.append(player_index)

        return indices
    
    def tournament_should_end(self):
        active_players = self.get_active_player_indices()

        return (
            self.current_round_number
            >= self.number_of_rounds
            or len(active_players) <= 1
        )

    def start_next_round(self):
        if (
            self.current_table_round is not None
            and not self.current_table_round.is_over
        ):
            raise ValueError(
                "The current round is not finished"
            )

        if self.tournament_should_end():
            self.is_over = True
            return None

        active_indices = (
            self.get_active_player_indices()
        )

        # Find the next active starting player.
        while (
            self.starting_player_index
            not in active_indices
        ):
            self.starting_player_index = (
                self.starting_player_index + 1
            ) % len(self.players)

        # TableRound uses local player indices.
        starting_local_index = active_indices.index(
            self.starting_player_index
        )

        round_players = [
            self.players[player_index]
            for player_index in active_indices
        ]

        for player in round_players:
            player.reset_round()

        self.active_player_indices = active_indices

        self.current_table_round = TableRound(
            players=round_players,
            shoe=self.shoe,
            minimum_bet=self.minimum_bet,
            starting_player=starting_local_index + 1,
            hit_soft_17=self.hit_soft_17,
            max_hands=self.max_hands,
        )

        self.current_round_number += 1

        return self.current_table_round

    def get_bot_for_round_player(self, round_player_index):
        if round_player_index < 0 or round_player_index >= len(self.active_player_indices):
            raise ValueError("Player index is outside the range of valid players")

        tournament_index = self.active_player_indices[round_player_index]

        bot = self.bots[tournament_index]

        return bot

    def place_round_bets(self):
        if self.current_table_round is None:
            raise TypeError("Table round must be of type TableRound")
        if self.current_table_round.is_over:
            raise ValueError("This round is already over")

        for round_player_index in self.current_table_round.betting_order:
            player = self.current_table_round.get_player(round_player_index)
            bot = self.get_bot_for_round_player(round_player_index)

            betting_observation = self.build_betting_observation(round_player_index)

            bet = bot.choose_bet(betting_observation)

            if isinstance(bet, bool) or not isinstance(bet, (int, float)):
                raise TypeError("Bet must be a number")
            if not math.isfinite(bet):
                raise ValueError("Bet must be finite")

            self.current_table_round.place_bet(round_player_index, bet)

        if not self.current_table_round.betting_complete():
            raise ValueError("Bets have not all been placed")

    def begin_player_actions(self):
        if self.current_table_round is None:
            raise ValueError("Player round does not exist")
        if not self.current_table_round.betting_complete():
            raise ValueError("betting is not complete")

        self.current_table_round.deal_initial_cards()
        naturals_ended =self.current_table_round.resolve_naturals()

        return naturals_ended

    def get_active_hand_index(self, round_player_index):
        active_hand_index = None

        player = self.current_table_round.get_player(round_player_index)

        for i in range(len(player.hands)):
            player_hand = player.get_hand(i)
            if not player_hand.is_settled and not player_hand.is_finished():
                active_hand_index = i
                return active_hand_index

        return active_hand_index

    def play_all_players(self):
        if self.current_table_round is None:
            raise ValueError("Table round does not exist")
        if not self.current_table_round.naturals_checked:
            raise ValueError("Naturals have not been checked")
        if self.current_table_round.is_over:
            raise ValueError("Round is already over")

        while True:
            round_player_index = self.current_table_round.get_current_player_index()

            if round_player_index is None:
                break

            hand_index = self.get_active_hand_index(round_player_index)

            if hand_index is None:
                self.current_table_round.advance_action_turn()
                continue

            player = self.current_table_round.get_player(round_player_index)

            player_hand = player.get_hand(hand_index)

            bot = self.get_bot_for_round_player(round_player_index)

            legal_actions = self.current_table_round.get_legal_actions(round_player_index, hand_index)

            action_observation = self.build_action_observation(round_player_index, hand_index)
            chosen_action = bot.choose_action(action_observation)

            if chosen_action not in legal_actions:
                raise ValueError("Chosen action is not legal")

            if chosen_action == Action.HIT:
                self.current_table_round.player_hit(round_player_index, hand_index)
            elif chosen_action == Action.STAND:
                self.current_table_round.player_stand(round_player_index, hand_index)
            elif chosen_action == Action.DOUBLE:
                additional_bet = min(player_hand.bet, player.bankroll)
                self.current_table_round.player_double(round_player_index, hand_index, additional_bet)
            elif chosen_action == Action.SPLIT:
                self.current_table_round.player_split(round_player_index, hand_index)
            elif chosen_action == Action.SURRENDER:
                self.current_table_round.player_surrender(round_player_index, hand_index)
            else:
                raise ValueError("Unknown action")


    def finish_current_round(self):
        if self.current_table_round is None:
            raise ValueError("The current round does not exist")

        table_round = self.current_table_round

        if table_round.is_over:
            outcomes = table_round.outcomes
        else:
            if table_round.get_current_player_index() is not None:
                raise ValueError("The current round is not over yet")

            table_round.play_dealer()

            outcomes = table_round.settle_round()

        self.round_history.append(table_round)

        self.current_table_round = None

        self.starting_player_index = (self.starting_player_index + 1) % len(self.players)

        if self.tournament_should_end():
            self.is_over = True

        return outcomes

    def play_one_round(self):
        table_round = self.start_next_round()

        if table_round is None:
            return None

        self.place_round_bets()

        naturals_ended = self.begin_player_actions()

        if not naturals_ended:
            self.play_all_players()

        outcomes = self.finish_current_round()

        return outcomes

    def get_rankings(self):
        rankings = sorted(
            [(player_index, player.bankroll) for player_index, player in enumerate(self.players)],
            key=lambda x: x[1],
            reverse=True,
        )
        return rankings

    def play_tournament(self):
        while not self.is_over:
            outcomes = self.play_one_round()

            if outcomes is None:
                self.is_over = True
                break


        return self.get_rankings()



    def build_betting_observation(self, round_player_index):
        if self.current_table_round is None:
            raise ValueError("The current round does not exist")

        table_round = self.current_table_round

        if table_round.betting_complete():
            raise ValueError("Betting is already complete")

        expected_player_index = table_round.betting_order[table_round.next_bettor_index]

        if round_player_index != expected_player_index:
            raise ValueError("It's not this player's turn to bet")

        permanent_player_index = self.active_player_indices[round_player_index]

        current_bets = tuple(0 for _ in range(len(self.players)))

        bets_placed = tuple(False for _ in range(len(self.players)))

        for local_index, player in enumerate(table_round.players):
            permanent_index = self.active_player_indices[local_index]
            if len(player.hands) > 0:
                first_hand = player.get_hand(0)
                current_bets = current_bets[:permanent_index] + (first_hand.bet,) + current_bets[permanent_index + 1:]
                bets_placed = bets_placed[:permanent_index] + (True,) + bets_placed[permanent_index + 1:]

        bankrolls = tuple(player.bankroll for player in self.players)

        active_players = tuple(
            player_index in self.active_player_indices
            for player_index in range(len(self.players)))


        permanent_betting_order = tuple(
            self.active_player_indices[local_index]
            for local_index in table_round.betting_order
        )

        return BettingObservation(
            round_number=self.current_round_number,
            total_rounds=self.number_of_rounds,
            rounds_remaining=self.number_of_rounds - self.current_round_number,
            player_index=permanent_player_index,
            round_player_index=round_player_index,
            betting_position=table_round.next_bettor_index,
            minimum_bet=self.minimum_bet,
            bankroll = self.players[permanent_player_index].bankroll,
            bankrolls=bankrolls,
            active_players=active_players,
            current_bets=current_bets,
            bets_placed=bets_placed,
            betting_order=permanent_betting_order,
        )

    def build_action_observation(self, round_player_index, hand_index):
        table_round = self.current_table_round

        if table_round is None:
            raise ValueError("no table_round")

        current_player_index = table_round.get_current_player_index()

        if current_player_index is None:
            raise ValueError("All players have finished acting")

        if round_player_index != current_player_index:
            raise ValueError("It is not this player's turn")

        player = table_round.get_player(round_player_index)

        player_hand = player.get_hand(hand_index)

        legal_actions = table_round.get_legal_actions(round_player_index, hand_index)

        if len(legal_actions) == 0:
            raise ValueError("No legal actions available")

        permanent_player_index = self.active_player_indices[round_player_index]

        bankrolls = tuple(player.bankroll for player in self.players)

        active_players = tuple(
            player_index in self.active_player_indices
            for player_index in range(len(self.players))
        )

        player_bets = list(0 for _ in range(len(self.players)))

        for local_index, local_player in enumerate(table_round.players):
            permanent_index = self.active_player_indices[local_index]

            total_bet = 0

            for local_player_hand in local_player.hands:
                total_bet += local_player_hand.bet

            player_bets[permanent_index] = total_bet

        player_bets = tuple(player_bets)

        current_bet = player_hand.bet

        hand_total = player_hand.hand.get_total()

        hand_is_soft = player_hand.hand.is_soft()

        hand_card_values = tuple(
            card.get_value()
            for card in player_hand.hand.cards
        )

        hand_came_from_split = player_hand.came_from_split

        dealer_upcard_value = table_round.dealer.hand.cards[0].get_value()

        permanent_betting_order = tuple(
            self.active_player_indices[local_index]
            for local_index in table_round.betting_order
        )

        ordered_legal_actions = tuple(
            action
            for action in Action
            if action in legal_actions
        )

        return ActionObservation(
            round_number=self.current_round_number,
            total_rounds=self.number_of_rounds,
            rounds_remaining=self.number_of_rounds - self.current_round_number,
            player_index=permanent_player_index,
            round_player_index=round_player_index,
            hand_index=hand_index,
            bankroll=self.players[permanent_player_index].bankroll,
            bankrolls=bankrolls,
            current_bet=current_bet,
            player_bets=player_bets,
            hand_total=hand_total,
            hand_is_soft=hand_is_soft,
            hand_card_values=hand_card_values,
            hand_came_from_split=hand_came_from_split,
            dealer_upcard_value=dealer_upcard_value,
            legal_actions=ordered_legal_actions,
            betting_order=permanent_betting_order,
            active_players=active_players,
            hit_soft_17=self.hit_soft_17
        )
