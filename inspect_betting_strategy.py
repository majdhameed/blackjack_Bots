from agents.basic_strategy_agent import BasicStrategyAgent
from agents.chasing_agent import ChasingAgent
from agents.controlled_lead_martingale_agent import (
    ControlledLeadMartingaleAgent,
)
from agents.early_lead_agent import EarlyLeadAgent
from agents.lead_protection_agent import (
    LeadProtectionAgent,
)
from agents.neural_betting_agent import (
    NeuralBettingAgent,
)
from ml.betting_encoder import (
    encode_betting_observation,
)
from ml.betting_network import (
    BETTING_ACTION_NAMES,
    BettingNetwork,
)
from tournament.observation import (
    BettingObservation,
)

from ml.betting_probes import (
    create_betting_probes,
)




def rank_and_gap_to_second(observation):
    active_bankrolls = [
        bankroll
        for bankroll, active in zip(
            observation.bankrolls,
            observation.active_players,
        )
        if active
    ]
    rank = 1 + sum(
        bankroll > observation.bankroll
        for bankroll in active_bankrolls
    )
    second_bankroll = sorted(
        active_bankrolls,
        reverse=True,
    )[1]
    gap_fraction = max(
        0,
        second_bankroll - observation.bankroll,
    ) / observation.bankroll
    return rank, gap_fraction


def inspect_strategy(network_path):
    network = BettingNetwork.load(network_path)
    agent = NeuralBettingAgent(network)
    minimum_agent = BasicStrategyAgent()
    chaser = ChasingAgent()
    controlled_lead = ControlledLeadMartingaleAgent()
    half_lead = EarlyLeadAgent(0.25, 0.50)
    all_in_lead = EarlyLeadAgent(0.10, 1.00)
    protector = LeadProtectionAgent()

    scenarios = create_betting_probes()

    print()
    print("Learned neural betting strategy")
    print("-------------------------------")

    header = (
        f"{'Scenario':<48}"
        f"{'Rank':>6}"
        f"{'Gap 2nd':>10}"
        f"{'Output':>10}"
        f"{'Action':>22}"
        f"{'Neural':>12}"
        f"{'Minimum':>12}"
        f"{'Chaser':>12}"
        f"{'Controlled':>12}"
        f"{'Half lead':>12}"
        f"{'All-in lead':>12}"
        f"{'Protector':>12}"
    )

    print(header)
    print("-" * len(header))

    minimum_matches = 0

    for scenario in scenarios:
        observation = scenario["observation"]
        rank, gap_fraction = rank_and_gap_to_second(
            observation
        )

        features = encode_betting_observation(
            observation
        )

        raw_fraction = network.forward(features)
        action_name = BETTING_ACTION_NAMES[
            network.preferred_action(features)
        ]

        final_bet = agent.choose_bet(
            observation
        )

        minimum_bet = minimum_agent.choose_bet(
            observation
        )
        chaser_bet = chaser.choose_bet(observation)
        controlled_bet = controlled_lead.choose_bet(
            observation
        )
        half_lead_bet = half_lead.choose_bet(
            observation
        )
        all_in_lead_bet = all_in_lead.choose_bet(
            observation
        )
        protector_bet = protector.choose_bet(
            observation
        )
        if final_bet == minimum_bet:
            minimum_matches += 1

        print(
            f"{scenario['name']:<48}"
            f"{rank:>6}"
            f"{gap_fraction:>9.1%}"
            f"{raw_fraction:>10.4f}"
            f"{action_name:>22}"
            f"{final_bet:>12,}"
            f"{minimum_bet:>12,}"
            f"{chaser_bet:>12,}"
            f"{controlled_bet:>12,}"
            f"{half_lead_bet:>12,}"
            f"{all_in_lead_bet:>12,}"
            f"{protector_bet:>12,}"
        )

    print()
    print(
        "Neural matches the minimum bettor in "
        f"{minimum_matches} of {len(scenarios)} scenarios."
    )


def main():
    inspect_strategy(
        "models/best_betting_network.npz"
    )


if __name__ == "__main__":
    main()
