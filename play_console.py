# Interactive single-player blackjack interface. Input/display concerns stay
# here while the blackjack package owns the game rules and state transitions.
import os

from blackjack.cards import Shoe
from blackjack.player import Player
from blackjack.round import Round


def ask_positive_integer(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Please enter a whole number greater than zero.")


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter yes or no.")


def format_money(amount):
    if float(amount).is_integer():
        return f"${int(amount):,}"
    return f"${amount:,.2f}"


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def display_cards(hand):
    return ", ".join(str(card) for card in hand.cards)


def display_player_hand(player_hand, hand_index):
    print(f"\nHand {hand_index + 1}")
    print(f"Cards: {display_cards(player_hand.hand)}")
    print(f"Total: {player_hand.hand.get_total()}")
    print(f"Bet: {format_money(player_hand.bet)}")

    if player_hand.hand.is_soft():
        print("This is a soft hand.")


def ask_bet(player, minimum_bet):
    # Keep prompting until the player quits or Player accepts the wager.
    while True:
        print(f"\nBankroll: {format_money(player.bankroll)}")
        print(f"Minimum bet: {format_money(minimum_bet)}")

        answer = input("Enter your bet, or q to quit: ").strip().lower()

        if answer in ("q", "quit", "exit"):
            return False

        try:
            bet = int(answer)
        except ValueError:
            print("Please enter a whole-number bet or q to quit.")
            continue

        try:
            player.place_bet(bet, minimum_bet)
            return True
        except ValueError as error:
            print(error)


def play_player_hands(round_):
    # The hand list can grow during this loop when the player splits.
    hand_index = 0

    while hand_index < len(round_.player.hands):
        player_hand = round_.player.get_hand(hand_index)

        if player_hand.is_finished():
            display_player_hand(player_hand, hand_index)
            print("This hand is finished.")
            hand_index += 1
            continue

        display_player_hand(player_hand, hand_index)
        print(f"Dealer shows: {round_.dealer.hand.cards[0]}")
        print("\nChoose an action:")
        print("1. Hit")
        print("2. Stand")
        print("3. Double")
        print("4. Split")
        print("5. Surrender")

        choice = input("> ").strip().lower()

        try:
            if choice in ("1", "hit", "h"):
                card = round_.player_hit(hand_index)
                print(f"You received: {card}")

            elif choice in ("2", "stand", "s"):
                round_.player_stand(hand_index)

            elif choice in ("3", "double", "d"):
                additional_bet = ask_positive_integer(
                    "Additional double-down bet: "
                )
                card = round_.player_double(
                    hand_index,
                    additional_bet,
                )
                print(f"You received: {card}")

            elif choice in ("4", "split", "p"):
                first_card, second_card = round_.player_split(
                    hand_index
                )
                print(
                    "Split complete. Replacement cards: "
                    f"{first_card} and {second_card}"
                )

            elif choice in ("5", "surrender", "r"):
                round_.player_surrender(hand_index)

            else:
                print("Choose hit, stand, double, split, or surrender.")

        except (ValueError, IndexError) as error:
            print(f"Action unavailable: {error}")


def display_results(round_):
    # Outcomes have the same ordering as the player's final hand list.
    print("\nDealer's hand")
    print(f"Cards: {display_cards(round_.dealer.hand)}")
    print(f"Total: {round_.dealer.hand.get_total()}")

    if round_.dealer.hand.is_bust():
        print("Dealer busted.")

    print("\nResults")
    for hand_index, player_hand in enumerate(round_.player.hands):
        display_player_hand(player_hand, hand_index)
        print(f"Outcome: {player_hand.outcome}")

    print(
        "\nFinal bankroll: "
        f"{format_money(round_.player.bankroll)}"
    )


def play_game():
    # Reuse the shoe across rounds until the game ends or it needs replacement.
    clear_console()
    print("Blackjack console game")

    bankroll = ask_positive_integer("Starting bankroll: ")
    deck_count = ask_positive_integer("Number of decks: ")
    minimum_bet = ask_positive_integer("Minimum bet: ")
    hit_soft_17 = ask_yes_no(
        "Does the dealer hit soft 17? (yes/no): "
    )

    player = Player(bankroll)
    shoe = Shoe(deck_count)

    while player.bankroll > 0:
        if shoe.cards_remaining() < 52:
            print("\nThe shoe is low. Shuffling a new shoe.")
            shoe = Shoe(deck_count)

        if not ask_bet(player, minimum_bet):
            print(
                "\nYou leave with "
                f"{format_money(player.bankroll)}."
            )
            return

        round_ = Round(
            player=player,
            shoe=shoe,
            minimum_bet=minimum_bet,
            hit_soft_17=hit_soft_17,
        )

        round_.deal_initial_cards()

        if not round_.resolve_naturals():
            play_player_hands(round_)
            round_.play_dealer()
            round_.settle_round()

        display_results(round_)
        player.reset_round()

        if player.bankroll > 0:
            input("\nPress Enter to start the next hand...")
            clear_console()

    print("\nYour bankroll is empty. Game over.")


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\nGame exited.")
