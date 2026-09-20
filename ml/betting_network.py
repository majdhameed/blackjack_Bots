import numpy as np


class BettingNetwork:
    def __init__(self, seed=None):
        if (
            seed is not None
            and not isinstance(seed, int)
        ):
            raise TypeError(
                "seed must be an integer or None"
            )

        self.random_generator = (
            np.random.default_rng(seed)
        )

        self.weights1 = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(32, 46),
            )
        )

        self.biases1 = np.zeros(32)

        self.weights2 = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(16, 32),
            )
        )

        self.biases2 = np.zeros(16)

        self.weights3 = (
            self.random_generator.normal(
                loc=0.0,
                scale=0.1,
                size=(1,16)
            )
        )

        self.biases3 = np.zeros(1)

    def forward(self, features):
        if len(features) != 46:
            raise ValueError("features must be of lenght 46")

        np_features = np.asarray(features, dtype=float)

        if not np.all(np.isfinite(np_features)):
            raise ValueError(
                "features must contain finite numbers"
            )

        hidden1 = self.weights1 @ np_features + self.biases1

        hidden1 = np.maximum(0, hidden1)

        hidden2 = self.weights2 @ hidden1 + self.biases2

        hidden2 = np.maximum(0, hidden2)

        output = self.weights3 @ hidden2 + self.biases3

        output = np.clip(output, -60, 60)

        output = 1 / (1 + np.exp(-output))

        return float(output.item())