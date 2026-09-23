from tournament.observation import BettingObservation


def validate_observation(observation):
    if not isinstance(observation, BettingObservation):
        raise TypeError(
            "observation must be a BettingObservation"
        )


def legal_bet(observation, requested_bet):
    """Clamp a wager to legal table-minimum chip increments."""
    validate_observation(observation)

    bankroll = observation.bankroll

    if bankroll <= 0:
        raise ValueError("No money")

    if bankroll <= observation.minimum_bet:
        return bankroll

    if requested_bet >= bankroll:
        return bankroll

    chip_increment = observation.minimum_bet
    requested_bet = max(
        chip_increment,
        requested_bet,
    )
    requested_bet = (
        int(requested_bet // chip_increment)
        * chip_increment
    )

    return int(
        min(
            bankroll,
            requested_bet,
        )
    )


def active_opponent_bankrolls(observation):
    validate_observation(observation)

    return [
        bankroll
        for player_index, bankroll in enumerate(
            observation.bankrolls
        )
        if (
            player_index != observation.player_index
            and observation.active_players[player_index]
        )
    ]


def top_two_cutoff(observation):
    active_bankrolls = sorted(
        (
            bankroll
            for player_index, bankroll in enumerate(
                observation.bankrolls
            )
            if observation.active_players[player_index]
        ),
        reverse=True,
    )

    if len(active_bankrolls) < 2:
        return active_bankrolls[0]

    return active_bankrolls[1]
