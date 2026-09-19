from dataclasses import dataclass

@dataclass(frozen=True)
class BettingObservation:
    round_number: int
    total_rounds: int
    rounds_remaining: int
    player_index: int
    round_player_index: int
    betting_position: int
    minimum_bet: int
    bankroll: float
    bankrolls: tuple[float]
    active_players: tuple[bool]
    current_bets: tuple[float]
    bets_placed: tuple[bool]
    betting_order : tuple[int]

@dataclass(frozen=True)
class ActionObservation:
    round_number: int
    total_rounds: int
    rounds_remaining: int
    player_index: int
    round_player_index: int
    hand_index: int
    bankroll: float
    bankrolls: tuple
    current_bet: float
    player_bets: tuple
    hand_total: int
    hand_is_soft: bool
    hand_card_values: tuple
    hand_came_from_split: bool
    dealer_upcard_value: int
    legal_actions: tuple
    betting_order: tuple
    active_players: tuple
    hit_soft_17: bool

