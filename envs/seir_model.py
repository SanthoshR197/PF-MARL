import numpy as np


class SEIRModel:
    """
    District-level stochastic SEIR model.
    """

    def __init__(
        self,
        beta=0.3,     # transmission rate
        sigma=0.2,    # exposed -> infected
        gamma=0.1,    # infected -> recovered
        noise_scale=0.01
    ):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.noise_scale = noise_scale

    def initialize(self, num_districts, initial_infected=1):
        """
        Initialize S, E, I, R for each district.
        """
        S = np.ones(num_districts)
        E = np.zeros(num_districts)
        I = np.zeros(num_districts)
        R = np.zeros(num_districts)

        # Seed infection in a few districts
        infected_indices = np.random.choice(
            num_districts, size=initial_infected, replace=False
        )

        for idx in infected_indices:
            S[idx] -= 0.01
            I[idx] += 0.01

        return S, E, I, R

    def step(self, S, E, I, R, mobility_matrix, health_measures=None):
        """
        Perform one SEIR update step.
        health_measures: Array (0.0 to 1.0) indicating mask/testing compliance per district.
        """
        num_districts = len(S)

        new_E = np.zeros(num_districts)
        new_I = np.zeros(num_districts)
        new_R = np.zeros(num_districts)

        for i in range(num_districts):
            # Infection pressure from neighbors
            infection_pressure = np.sum(mobility_matrix[i] * I)

            # Evaluate effective transmission rate
            # High health measure (e.g. 1.0) blocks up to 80% of transmissions (masks + tests)
            local_compliance = health_measures[i] if health_measures is not None else 0.0
            effective_beta = self.beta * (1.0 - 0.8 * local_compliance)

            # S -> E
            new_E[i] = (
                effective_beta
                * S[i]
                * infection_pressure
                + np.random.normal(0, self.noise_scale)
            )

            # E -> I
            new_I[i] = self.sigma * E[i]

            # I -> R
            new_R[i] = self.gamma * I[i]

        # Update compartments
        S = np.clip(S - new_E, 0, 1)
        E = np.clip(E + new_E - new_I, 0, 1)
        I = np.clip(I + new_I - new_R, 0, 1)
        R = np.clip(R + new_R, 0, 1)

        return S, E, I, R
