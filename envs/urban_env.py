import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

from envs.seir_model import SEIRModel


class UrbanEpidemicEnv(gym.Env):
    """
    Real-world aligned synthetic urban environment for PF-MARL.
    Represents a city as interconnected administrative districts
    with SEIR disease dynamics.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_districts=12, seed=42):
        super().__init__()

        self.num_districts = num_districts
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # -----------------------------
        # Build city graph
        # -----------------------------
        self.graph = self._build_city_graph()

        # -----------------------------
        # SEIR disease model
        # -----------------------------
        self.seir = SEIRModel()

        self.S, self.E, self.I, self.R = self.seir.initialize(
            self.num_districts, initial_infected=1
        )

        # Mobility matrix derived from graph
        self.mobility_matrix = self._build_mobility_matrix()

        # -----------------------------
        # Observation & action spaces
        # -----------------------------
        # Features per district:
        # population, mobility, economic weight, infection level
        self.features_per_node = 4

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_districts * self.features_per_node,),
            dtype=np.float32
        )

        # Policy intensity signal per district:
        # [0:num_districts] = Lockdown level
        # [num_districts:2*num_districts] = Health Measures (Masks/Testing)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_districts * 2,),
            dtype=np.float32
        )

        self.state = None

    # ------------------------------------------------------------------
    # City construction
    # ------------------------------------------------------------------
    def _build_city_graph(self):
        G = nx.Graph()

        district_types = ["residential", "commercial", "medical"]

        for i in range(self.num_districts):
            d_type = self.rng.choice(
                district_types,
                p=[0.6, 0.3, 0.1]  # realistic city composition
            )

            G.add_node(
                i,
                district_type=d_type,
                population=self.rng.uniform(0.4, 1.0),
                base_mobility=self.rng.uniform(0.3, 0.9),
                economic_weight=(
                    1.0 if d_type == "commercial"
                    else 0.6 if d_type == "medical"
                    else 0.4
                ),
            )

        # Ensure basic connectivity
        for i in range(self.num_districts - 1):
            G.add_edge(
                i,
                i + 1,
                mobility_weight=self.rng.uniform(0.4, 1.0)
            )

        # Add additional realistic links
        extra_edges = self.num_districts // 2
        for _ in range(extra_edges):
            a, b = self.rng.choice(self.num_districts, size=2, replace=False)
            G.add_edge(
                a,
                b,
                mobility_weight=self.rng.uniform(0.2, 0.8)
            )

        return G

    # ------------------------------------------------------------------
    # Mobility matrix
    # ------------------------------------------------------------------
    def _build_mobility_matrix(self):
        """
        Build normalized mobility matrix from graph.
        """
        M = np.zeros((self.num_districts, self.num_districts))

        for i, j, data in self.graph.edges(data=True):
            w = data["mobility_weight"]
            M[i, j] = w
            M[j, i] = w

        # Normalize rows to avoid explosion
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = M / row_sums

        return M

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self):
        obs = []
        for node_id, data in self.graph.nodes(data=True):
            obs.extend([
                data["population"],
                data["base_mobility"],
                data["economic_weight"],
                self.I[node_id],   # infection level
            ])
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reinitialize SEIR
        self.S, self.E, self.I, self.R = self.seir.initialize(
            self.num_districts, initial_infected=1
        )

        self.state = self._get_observation()
        return self.state, {}

    def step(self, action):
        """
        One simulation step = one time unit (e.g., one day).
        """
        action = np.clip(action, 0.0, 1.0)

        # Unpack actions
        lockdowns = action[:self.num_districts]
        health_measures = action[self.num_districts:]

        # -----------------------------
        # Policy effect on mobility (with inertia)
        # -----------------------------
        for i, a in enumerate(lockdowns):
            node = self.graph.nodes[i]
            node["base_mobility"] = np.clip(
                0.9 * node["base_mobility"] + 0.1 * (1 - a),
                0.1,
                1.0
            )

        # -----------------------------
        # SEIR disease update
        # -----------------------------
        self.S, self.E, self.I, self.R = self.seir.step(
            self.S, self.E, self.I, self.R, self.mobility_matrix, health_measures
        )

        self.state = self._get_observation()

        # Calculate Economic Penalty
        # 1. Cost of restricting mobility (businesses shut down)
        # 2. Cost of deploying health measures (buying masks, running testing centers)
        economic_cost = 0.0
        for i, node_data in self.graph.nodes(data=True):
            mobility_drop = max(0, 1.0 - node_data["base_mobility"])
            health_cost = health_measures[i] * 0.1  # Health measures are 10x cheaper than lockdown
            economic_cost += (mobility_drop + health_cost) * node_data["economic_weight"]

        # Dual-Objective Reward: minimize total infections AND economic cost
        alpha = 2.0  # Weight for lives saved
        beta = 0.5   # Weight for economy
        # Exponential penalty for infections to force the AI to crush virus peaks
        reward = - (alpha * (np.sum(self.I) ** 2.0) + beta * economic_cost)
        
        terminated = False
        truncated = False
        info = {
            'infections': np.sum(self.I) * 100,
            'economic_cost': economic_cost
        }

        return self.state, reward, terminated, truncated, info

    def render(self):
        pass
