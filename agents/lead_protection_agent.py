from agents.basic_strategy_agent import BasicStrategyAgent
from agents.betting_strategy_helpers import (
    active_opponent_bankrolls,
    legal_bet,
    top_two_cutoff,
)


class LeadProtectionAgent(BasicStrategyAgent):
    """Build a useful early lead, then cover visible threats."""

    def __init__(
        self,
        safe_lead_fraction=0.10,
        lead_build_fraction=0.20,
        build_until_fraction=0.67,
    ):
        for name, value in (
            ("safe_lead_fraction", safe_lead_fraction),
            ("lead_build_fraction", lead_build_fraction),
            ("build_until_fraction", build_until_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between zero and one"
                )

        if lead_build_fraction == 0:
            raise ValueError(
                "lead_build_fraction must be greater than zero"
            )

        self.safe_lead_fraction = safe_lead_fraction
        self.lead_build_fraction = lead_build_fraction
        self.build_until_fraction = build_until_fraction

    def choose_bet(self, observation):
        opponents = active_opponent_bankrolls(
            observation
        )

        highest_opponent = max(opponents, default=0)
        is_leading = (
            not opponents
            or observation.bankroll > highest_opponent
        )

        requested_bet = observation.minimum_bet

        if is_leading:
            # Earlier bettors have already had their wager removed from
            # bankroll. A win returns twice that visible wager.
            largest_winning_bankroll = max(
                (
                    bankroll + 2 * observation.current_bets[index]
                    for index, bankroll in enumerate(
                        observation.bankrolls
                    )
                    if (
                        index != observation.player_index
                        and observation.active_players[index]
                        and observation.bets_placed[index]
                    )
                ),
                default=observation.bankroll,
            )

            requested_bet = max(
                observation.minimum_bet,
                largest_winning_bankroll
                - observation.bankroll
                + 1,
            )

            lead_margin = (
                observation.bankroll - highest_opponent
            )
            progress = (
                observation.round_number
                / observation.total_rounds
            )
            if (
                opponents
                and progress <= self.build_until_fraction
                and lead_margin
                < highest_opponent * self.safe_lead_fraction
            ):
                requested_bet = max(
                    requested_bet,
                    observation.bankroll
                    * self.lead_build_fraction,
                )
        elif (
            observation.rounds_remaining
            <= max(1, observation.total_rounds // 4)
        ):
            deficit = (
                top_two_cutoff(observation)
                - observation.bankroll
            )

            requested_bet = min(
                observation.bankroll * 0.35,
                observation.minimum_bet
                + max(0, deficit),
            )

        return legal_bet(
            observation,
            requested_bet,
        )
