# Blackjack Bots

Blackjack Bots is a Python project for modeling blackjack rules, running games, and comparing betting strategies over repeated tournaments. The current focus is a reusable game engine plus command-line simulations. A future visualization layer can turn simulation results into charts that make strategy performance easier to compare.

## Project scope

The project currently covers three related use cases:

- Play blackjack interactively in a terminal.
- Simulate a basic-strategy player over many rounds.
- Run tournaments in which several bots use the same card-play strategy but different betting systems.

The engine handles normal blackjack actions and round settlement, including hits, stands, doubles, splits, surrender, dealer blackjack, natural blackjack, pushes, configurable soft-17 behavior, multiple decks, and a maximum number of split hands.

This project is intended for simulation and software experimentation. It is not gambling advice, and past simulated performance does not predict real-world results.

## Betting strategies

The tournament simulator currently compares seven strategies:

| Strategy | Wager |
| --- | --- |
| Minimum | The table minimum, or the remaining bankroll when it is lower |
| 10% | 10% of the current bankroll, subject to the table minimum |
| 20% | 20% of the current bankroll, subject to the table minimum |
| 30% | 30% of the current bankroll, subject to the table minimum |
| 40% | 40% of the current bankroll, subject to the table minimum |
| 50% | 50% of the current bankroll, subject to the table minimum |
| All-in | The entire current bankroll |

Every included bot inherits the same basic-strategy card decisions. This isolates bet sizing as the main difference between competitors.

Strategies are randomly assigned to seats for every tournament to reduce seat-order bias. A tournament ends when it reaches the requested round limit or no more than one funded player remains.

## Project structure

```text
blackjack_Bots/
|-- agents/                 Betting and playing strategies
|-- blackjack/              Cards, hands, players, rules, and settlement
|-- ml/                     Placeholder for future model feature encoding
|-- tournament/             Multi-player rounds and tournament coordination
|-- tests/                  Automated rule and tournament tests
|-- play_console.py         Interactive terminal game
|-- simulate.py             Single-player simulation driver
`-- simulate_tournament.py  Seven-strategy comparison
```

The main layers are:

- `blackjack`: Owns the core game rules and mutable state.
- `tournament`: Coordinates multiple players, betting order, action order, elimination, and rankings.
- `agents`: Selects bets and actions from immutable observations supplied by the tournament.
- Entry-point scripts: Collect terminal input and present simulation results.

The observation objects intentionally keep bots separate from mutable engine objects. This makes strategy code safer and provides a useful boundary for future machine-learning agents.

## Requirements

- Python 3.9 or newer
- `pytest` to run the test suite

The game and simulation code use only the Python standard library.

## Running the project

Run commands from the repository root.

### Interactive blackjack

```powershell
python play_console.py
```

The console prompts for a starting bankroll, deck count, minimum bet, and the dealer's soft-17 rule.

### Strategy tournament

```powershell
python simulate_tournament.py
```

The simulator prompts for:

- Starting bankroll
- Hands (rounds) requested per tournament
- Number of tournaments
- Number of decks
- Minimum bet
- Whether the dealer hits soft 17

It reports first-place frequency, tie-adjusted win rate, average ending bankroll, bankruptcy rate, and hand outcome counts for each strategy.

### Single-player simulation

```powershell
python simulate.py
```

This script is the older single-player simulation path. Its agent calls still need to be aligned fully with the newer observation-based bot interface, so the tournament simulator is currently the primary automated entry point.

## Running tests

```powershell
python -m pytest -q
```

The tests cover cards, hand totals, basic strategy, bankroll operations, legal actions, splitting aces, settlement, table rounds, card counting, and tournament behavior.

## Current limitations

- Results are printed to the terminal rather than saved as structured data.
- Simulations do not currently accept a fixed random seed from the command line.
- The single-player simulator predates the observation-based agent API.
- The card counter is implemented and tested but is not yet connected to a betting bot.
- The `ml` package is only a placeholder for future observation encoding.
- There is no graphical interface, report export, or chart generation yet.

## Roadmap

The next reporting milestone is to store simulation results and visualize strategy behavior. Useful additions include:

1. Export each run to CSV or JSON.
2. Add reproducible random seeds and configuration metadata.
3. Plot average ending bankroll by strategy.
4. Compare first-place and bankruptcy rates.
5. Chart bankroll distributions instead of averages alone.
6. Show bankroll trajectories over the course of a tournament.
7. Add confidence intervals as the number of simulations grows.
8. Connect the card counter and future learned agents to the observation API.

Separating result collection from chart rendering will make it possible to support notebooks, static image reports, or a small dashboard without changing the blackjack engine.
