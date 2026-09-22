import copy
from pathlib import Path

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

        self.biases3 = np.array([-8.0])

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

    def clone(self):

        cloned_net = BettingNetwork()

        cloned_net.random_generator.bit_generator.state = (
            copy.deepcopy(
                self.random_generator.bit_generator.state
            )
        )

        cloned_net.weights1 = np.copy(self.weights1)
        cloned_net.biases1 = np.copy(self.biases1)
        
        cloned_net.weights2 = np.copy(self.weights2)
        cloned_net.biases2 = np.copy(self.biases2)
        
        cloned_net.weights3 = np.copy(self.weights3)
        cloned_net.biases3 = np.copy(self.biases3)
        
        cloned_net.random_generator = np.random.default_rng()
        cloned_net.random_generator.bit_generator.state = self.random_generator.bit_generator.state
        
        return cloned_net

    def mutate(self, mutation_rate, mutation_strength):

        if not (0 <= mutation_rate <= 1):
            raise ValueError("mutation rate must be between 0 and 1")

        if mutation_strength < 0:
            raise ValueError("mutation strength must be a postive number")

        parameter_names = (
            "weights1",
            "biases1",
            "weights2",
            "biases2",
            "weights3",
            "biases3",
        )

        for parameter_name in parameter_names:
            parameter = getattr(self, parameter_name)

            mutation_mask = (
                self.random_generator.random(
                     size=parameter.shape
                )
                < mutation_rate
            )

            mutation_noise = (
                self.random_generator.normal(
                    loc=0.0,
                    scale=mutation_strength,
                    size=parameter.shape,
                )
            )

            parameter += mutation_mask * mutation_noise
    def get_parameters(self):
        return {
            "weights1": self.weights1.copy(),
            "biases1": self.biases1.copy(),
            "weights2": self.weights2.copy(),
            "biases2": self.biases2.copy(),
            "weights3": self.weights3.copy(),
            "biases3": self.biases3.copy(),
        }
    
    def save(self, file_path):
        file_path = Path(file_path)

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            file_path,
            **self.get_parameters(),
        )


    @classmethod
    def load(cls, file_path):
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Network file does not exist: {file_path}"
            )

        expected_shapes = {
            "weights1": (32, 46),
            "biases1": (32,),
            "weights2": (16, 32),
            "biases2": (16,),
            "weights3": (1, 16),
            "biases3": (1,),
        }

        with np.load(
            file_path,
            allow_pickle=False,
        ) as data:
            missing_parameters = (
                set(expected_shapes) - set(data.files)
            )

            if missing_parameters:
                raise ValueError(
                    "Network file is missing parameters: "
                    f"{sorted(missing_parameters)}"
                )

            loaded_parameters = {}

            for parameter_name, expected_shape in (
                expected_shapes.items()
            ):
                parameter = np.asarray(
                    data[parameter_name],
                    dtype=float,
                )

                if parameter.shape != expected_shape:
                    raise ValueError(
                        f"{parameter_name} must have shape "
                        f"{expected_shape}, but has shape "
                        f"{parameter.shape}"
                    )

                if not np.all(np.isfinite(parameter)):
                    raise ValueError(
                        f"{parameter_name} contains "
                        "non-finite values"
                    )

                loaded_parameters[
                    parameter_name
                ] = parameter.copy()

        network = cls()

        network.weights1 = loaded_parameters[
            "weights1"
        ]

        network.biases1 = loaded_parameters[
            "biases1"
        ]

        network.weights2 = loaded_parameters[
            "weights2"
        ]

        network.biases2 = loaded_parameters[
            "biases2"
        ]

        network.weights3 = loaded_parameters[
            "weights3"
        ]

        network.biases3 = loaded_parameters[
            "biases3"
        ]

        return network