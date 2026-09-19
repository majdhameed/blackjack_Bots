# Batch simulator for one automated player. It drives the single-player Round
# engine repeatedly and aggregates wagers, outcomes, and ending bankroll.
from collections import Counter

from agents.basic_strategy_agent import BasicStrategyAgent
from blackjack.actions import Action
from blackjack.cards import Shoe
from blackjack.player import Player
from blackjack.round import Round


def execute_action(
    round_,
    bot,
    hand_index,
):
    action = bot.choose_action(
        round_,
        hand_index,
    )

    # Translate the bot's selected enum into the corresponding engine command.
    if action == Action.HIT:
        round_.player_hit(hand_index)

    elif action == Action.STAND:
        round_.player_stand(hand_index)

    elif action == Action.DOUBLE:
        player_hand = round_.player.get_hand(
            hand_index
        )

        additional_bet = min(
            player_hand.bet,
            round_.player.bankroll,
        )

        round_.player_double(
            hand_index,
            additional_bet,
        )

    elif action == Action.SPLIT:
        round_.player_split(hand_index)

    elif action == Action.SURRENDER:
        round_.player_surrender(hand_index)

    else:
        raise ValueError(
            f"Unknown action: {action}"
        )


def play_player_hands(round_, bot):
    # Splitting can append hands, so the outer loop checks the live list length.
    hand_index = 0
    action_count = 0

    while hand_index < len(round_.player.hands):
        player_hand = round_.player.get_hand(
            hand_index
        )

        while not player_hand.is_finished():
            legal_actions = (
                round_.get_legal_actions(
                    hand_index
                )
            )

            if not legal_actions:
                raise RuntimeError(
                    "Active hand has no legal actions"
                )

            execute_action(
                round_,
                bot,
                hand_index,
            )

            # Guarding the count later prevents a faulty bot from looping forever.
            action_count += 1

            if action_count > 100:
                raise RuntimeError(
                    "Too many actions in one round"
                )

            # Splitting replaces the current hand,
            # so retrieve it again.
            player_hand = (
                round_.player.get_hand(
                    hand_index
                )
            )

        hand_index += 1


def play_single_round(
    player,
    shoe,
    bot,
    minimum_bet,
    hit_soft_17,
    max_hands=4,
):
    # Each round follows the same phases: wager, deal, player actions, dealer,
    # then settlement.
    bet = bot.choose_bet(
        player,
        minimum_bet,
    )

    player.place_bet(
        bet,
        minimum_bet,
    )

    round_ = Round(
        player=player,
        shoe=shoe,
        minimum_bet=minimum_bet,
        hit_soft_17=hit_soft_17,
        max_hands=max_hands,
    )

    round_.deal_initial_cards()

    natural_ended_round = (
        round_.resolve_naturals()
    )

    if not natural_ended_round:
        play_player_hands(round_, bot)
        round_.play_dealer()
        round_.settle_round()

    return round_


def simulate(
    number_of_rounds,
    starting_bankroll,
    deck_count,
    minimum_bet,
    hit_soft_17,
):
    player = Player(starting_bankroll)
    bot = BasicStrategyAgent()
    shoe = Shoe(deck_count)

    outcome_counts = Counter()

    rounds_played = 0
    total_hands = 0
    total_wagered = 0

    # Stop early if the player loses the entire bankroll.
    for _ in range(number_of_rounds):
        if player.bankroll <= 0:
            break

        # For now, reshuffle between rounds when
        # fewer than 52 cards remain.
        if shoe.cards_remaining() < 52:
            shoe = Shoe(deck_count)

        round_ = play_single_round(
            player=player,
            shoe=shoe,
            bot=bot,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
        )

        rounds_played += 1
        total_hands += len(player.hands)

        total_wagered += sum(
            player_hand.bet
            for player_hand in player.hands
        )

        outcome_counts.update(
            round_.outcomes
        )

        player.reset_round()

    ending_bankroll = player.bankroll
    net_result = (
        ending_bankroll - starting_bankroll
    )

    if total_wagered > 0:
        return_percentage = (
            net_result / total_wagered
        ) * 100
    else:
        return_percentage = 0

    return {
        "rounds_requested": number_of_rounds,
        "rounds_played": rounds_played,
        "total_hands": total_hands,
        "total_wagered": total_wagered,
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": ending_bankroll,
        "net_result": net_result,
        "return_percentage": return_percentage,
        "outcomes": outcome_counts,
    }


def print_results(results):
    # Keep presentation separate from simulation so callers can reuse the data.
    print("\nSimulation results")
    print("------------------")

    print(
        "Rounds requested:",
        f"{results['rounds_requested']:,}",
    )

    print(
        "Rounds played:",
        f"{results['rounds_played']:,}",
    )

    print(
        "Total hands:",
        f"{results['total_hands']:,}",
    )

    print(
        "Total wagered:",
        f"${results['total_wagered']:,.2f}",
    )

    print(
        "Starting bankroll:",
        f"${results['starting_bankroll']:,.2f}",
    )

    print(
        "Ending bankroll:",
        f"${results['ending_bankroll']:,.2f}",
    )

    print(
        "Net result:",
        f"${results['net_result']:,.2f}",
    )

    print(
        "Return on amount wagered:",
        f"{results['return_percentage']:.4f}%",
    )

    print("\nOutcomes")
    print("--------")

    for outcome, count in sorted(
        results["outcomes"].items()
    ):
        print(
            f"{outcome}: {count:,}"
        )


def ask_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))

            if value <= 0:
                raise ValueError

            return value

        except ValueError:
            print(
                "Please enter a whole number "
                "greater than zero."
            )


def main():
    starting_bankroll = ask_positive_integer(
        "Enter the starting bankroll: "
    )

    number_of_rounds = ask_positive_integer(
        "How many rounds would you like to simulate? "
    )

    results = simulate(
        number_of_rounds=number_of_rounds,
        starting_bankroll=starting_bankroll,
        deck_count=6,
        minimum_bet=10,
        hit_soft_17=True,
    )

    print_results(results)

if __name__ == "__main__":
    main()
