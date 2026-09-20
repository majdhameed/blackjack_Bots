# Reserved for converting BettingObservation values into model-ready numeric features. 
from tournament.observation import BettingObservation

def encode_betting_observation(observation: BettingObservation) -> tuple[float, ...]:
    if not isinstance(observation, BettingObservation):
        raise TypeError("Observation must be a betting observation")

    player_count = len(observation.bankrolls)

    if player_count != 7:
        raise ValueError("player count incorrect")

    if len(observation.active_players) != 7:
        raise ValueError("active_players must contain 7 values")

    if len(observation.current_bets) != 7:
        raise ValueError("current_bets must contain 7 values")

    if len(observation.bets_placed) != 7:
        raise ValueError("bets_placed must contain 7 values")

    if len(observation.card_value_counts) != 10:
        raise ValueError(
            "card_value_counts must contain 10 values"
        )

    if observation.total_rounds <= 0:
        raise ValueError(
            "total_rounds must be positive"
        )

    money_scale = max(max(observation.bankrolls), observation.minimum_bet, 1)

    features = [
        observation.round_number / observation.total_rounds,
        observation.rounds_remaining / observation.total_rounds,
        observation.bankroll / money_scale,
        observation.minimum_bet / money_scale,
        observation.betting_position / 6
    ]

    features.extend(
        bankroll / money_scale
        for bankroll in observation.bankrolls
    )

    features.extend(
        1.0 if active else 0.0
        for active in observation.active_players
    )

    features.extend(
        bet / money_scale
        for bet in observation.current_bets
    )

    features.extend(
        1.0 if bet_placed else 0.0
        for bet_placed in observation.bets_placed
    )

    visible_card_scaled = max(observation.cards_seen, 1)

    features.extend(
        card_count / visible_card_scaled
        for card_count in observation.card_value_counts
    )

    features.append(observation.running_count / visible_card_scaled)

    features.append(max(-10, min(observation.true_count, 10))/10)

    features.append(observation.shoe_penetration)

    if len(features) != 46:
        raise ValueError(
            f"Expected 46 features, got {len(features)}"
        )
    return tuple(float(value) for value in features)

    