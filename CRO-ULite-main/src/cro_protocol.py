"""
cro_protocol.py
===============
CRO-Ulite protocol compatible with Graph-based WSN simulation.

Pipeline per round:
    1) Build alive-node feature matrix
    2) PCA + K-means candidate filtering
    3) Coral Reef Optimization (CRO) for CH subset selection
    4) Assign each alive non-CH node to nearest CH
    5) Update energy:
         - member -> CH transmission
         - CH reception
         - CH aggregation
         - CH -> BS transmission
"""

import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.energy_model import (
    calculate_transmission_energy,
    calculate_reception_energy,
    calculate_aggregation_energy,
    update_energy,
)
from src.network_setup import BASE_STATION


# =========================================================
# Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# Helper Functions
# =========================================================
def euclidean_distance(p1, p2):
    """Return Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_alive_nodes(nodes, G):
    """Return alive node IDs only."""
    return [node for node in nodes if G.nodes[node]["energy"] > 0]


def safe_update_energy(node, energy_spent, G):
    """Update node energy and clamp to zero."""
    update_energy(node, energy_spent, G)
    if G.nodes[node]["energy"] < 0:
        G.nodes[node]["energy"] = 0


# =========================================================
# Fitness Function
# =========================================================
def compute_fitness(candidate_chs, alive_nodes, positions, G, base_station):
    """
    Evaluate a candidate CH subset.

    Fitness aims to:
    - maximize average CH residual energy
    - minimize average member-to-CH distance
    - minimize average CH-to-BS distance
    - softly regularize CH count
    """
    if not candidate_chs:
        return -np.inf

    max_energy = max(G.nodes[node]["energy"] for node in alive_nodes) if alive_nodes else 1.0
    if max_energy <= 0:
        return -np.inf

    # 1) Average normalized CH energy
    avg_ch_energy = np.mean([G.nodes[ch]["energy"] for ch in candidate_chs]) / max_energy

    # 2) Average member-to-nearest-CH distance
    member_nodes = [node for node in alive_nodes if node not in candidate_chs]
    if member_nodes:
        member_distances = [
            min(euclidean_distance(positions[node], positions[ch]) for ch in candidate_chs)
            for node in member_nodes
        ]
        avg_member_dist = np.mean(member_distances) / 141.4
    else:
        avg_member_dist = 0.0

    # 3) Average CH-to-BS distance
    ch_bs_distances = [
        euclidean_distance(positions[ch], base_station) for ch in candidate_chs
    ]
    avg_ch_bs_dist = np.mean(ch_bs_distances) / 200.0

    # 4) Soft regularization for CH count
    target_ratio = 0.2
    target_count = max(1, int(len(alive_nodes) * target_ratio))
    count_penalty = abs(len(candidate_chs) - target_count) / max(1, target_count)

    fitness = (
        0.4 * avg_ch_energy
        - 0.25 * avg_member_dist
        - 0.25 * avg_ch_bs_dist
        - 0.10 * count_penalty
    )

    return fitness


# =========================================================
# Stage 1: PCA + K-means Candidate Filtering
# =========================================================
def pca_kmeans_filter(alive_nodes, positions, G, base_station, ch_ratio=0.2, pca_components=2):
    """
    Build node features, scale them, apply PCA, then K-means.
    Select one high-energy candidate per cluster.
    """
    if not alive_nodes:
        return []

    if len(alive_nodes) == 1:
        return alive_nodes[:]

    bs_x, bs_y = base_station

    features = []
    for node in alive_nodes:
        x, y = positions[node]
        energy = G.nodes[node]["energy"]
        signal_strength = G.nodes[node].get("signal_strength", 0.0)
        dist_to_bs = euclidean_distance((x, y), (bs_x, bs_y))

        features.append([x, y, energy, signal_strength, dist_to_bs])

    features = np.array(features, dtype=float)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    n_comp = min(pca_components, features_scaled.shape[1], len(alive_nodes))
    if len(alive_nodes) < 2 or n_comp < 1:
        reduced = features_scaled
    else:
        pca = PCA(n_components=n_comp, random_state=SEED)
        reduced = pca.fit_transform(features_scaled)

    n_clusters = max(1, int(len(alive_nodes) * ch_ratio))
    n_clusters = min(n_clusters, len(alive_nodes))

    if len(alive_nodes) <= n_clusters:
        return alive_nodes[:]

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(reduced)

    candidates = []
    for cluster_id in range(n_clusters):
        cluster_members = [
            alive_nodes[i] for i, lbl in enumerate(labels) if lbl == cluster_id
        ]
        if cluster_members:
            best = max(cluster_members, key=lambda node: G.nodes[node]["energy"])
            candidates.append(best)

    return candidates


# =========================================================
# Stage 2: Coral Reef Optimization
# =========================================================
class CoralReefOptimizer:
    """CRO optimizer for selecting the best subset of CH candidates."""

    def __init__(
        self,
        candidates,
        alive_nodes,
        positions,
        G,
        base_station,
        reef_size=20,
        n_iterations=30,
        fa=0.1,
        fd=0.1,
        mutation_rate=0.3,
        k_attempts=3,
    ):
        self.candidates = candidates
        self.alive_nodes = alive_nodes
        self.positions = positions
        self.G = G
        self.base_station = base_station
        self.reef_size = min(reef_size, max(1, 2 ** min(len(candidates), 10)))
        self.n_iterations = n_iterations
        self.fa = fa
        self.fd = fd
        self.mutation_rate = mutation_rate
        self.k_attempts = k_attempts

    def _fitness(self, coral):
        return compute_fitness(
            candidate_chs=coral,
            alive_nodes=self.alive_nodes,
            positions=self.positions,
            G=self.G,
            base_station=self.base_station,
        )

    def _random_coral(self):
        if not self.candidates:
            return []
        k = random.randint(1, max(1, int(len(self.candidates))))
        return random.sample(self.candidates, min(k, len(self.candidates)))

    def _broadcast_spawning(self, coral_a, coral_b):
        combined = list(set(coral_a + coral_b))
        random.shuffle(combined)
        if not combined:
            return [random.choice(self.candidates)]
        mid = max(1, len(combined) // 2)
        return combined[:mid]

    def _budding(self, coral):
        if not self.candidates:
            return coral
        bud = coral.copy()
        extra = random.choice(self.candidates)
        if extra not in bud:
            bud.append(extra)
        return list(set(bud))

    def _mutate(self, coral):
        if not coral or not self.candidates:
            return coral
        mutated = coral.copy()
        idx = random.randrange(len(mutated))
        mutated[idx] = random.choice(self.candidates)
        return list(set(mutated))

    def _settle(self, reef, larva):
        larva_fitness = self._fitness(larva)
        for _ in range(self.k_attempts):
            idx = random.randrange(len(reef))
            if larva_fitness > self._fitness(reef[idx]):
                reef[idx] = larva
                return True
        return False

    def optimize(self):
        if not self.candidates:
            return []
        if len(self.candidates) == 1:
            return self.candidates[:]

        reef = [self._random_coral() for _ in range(self.reef_size)]
        best_coral = max(reef, key=self._fitness)
        best_fitness = self._fitness(best_coral)

        for _ in range(self.n_iterations):
            larvae = []

            random.shuffle(reef)
            for i in range(0, len(reef) - 1, 2):
                larva = self._broadcast_spawning(reef[i], reef[i + 1])
                if random.random() < self.mutation_rate:
                    larva = self._mutate(larva)
                larvae.append(larva)

            n_buds = max(1, int(self.fa * self.reef_size))
            top_corals = sorted(reef, key=self._fitness, reverse=True)[:n_buds]
            for coral in top_corals:
                bud = self._budding(coral)
                if random.random() < self.mutation_rate:
                    bud = self._mutate(bud)
                larvae.append(bud)

            for larva in larvae:
                if larva:
                    self._settle(reef, larva)

            n_remove = max(1, int(self.fd * self.reef_size))
            reef.sort(key=self._fitness, reverse=True)
            reef = reef[:max(1, len(reef) - n_remove)]

            while len(reef) < self.reef_size:
                reef.append(self._random_coral())

            current_best = max(reef, key=self._fitness)
            current_fitness = self._fitness(current_best)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_coral = current_best

        return best_coral if best_coral else self.candidates[:1]


# =========================================================
# Stage 3: Cluster Assignment
# =========================================================
def assign_clusters(alive_nodes, cluster_heads, positions):
    """Assign each alive non-CH node to nearest CH."""
    if not cluster_heads:
        return {}

    clusters = {ch: [] for ch in cluster_heads}

    for node in alive_nodes:
        if node in cluster_heads:
            continue

        nearest_ch = min(
            cluster_heads,
            key=lambda ch: euclidean_distance(positions[node], positions[ch])
        )
        clusters[nearest_ch].append(node)

    return clusters


# =========================================================
# Energy Update for One Round
# =========================================================
def update_cluster_energy(clusters, cluster_heads, positions, G, base_station):
    """
    Update energy for one simulation round:
    - member nodes transmit to CH
    - CH receives
    - CH aggregates
    - CH transmits to BS
    """
    # Member -> CH
    for ch, members in clusters.items():
        for member in members:
            if G.nodes[member]["energy"] <= 0:
                continue

            d_member_ch = euclidean_distance(positions[member], positions[ch])
            tx_energy = calculate_transmission_energy(d_member_ch)
            safe_update_energy(member, tx_energy, G)

    # CH reception + aggregation
    for ch, members in clusters.items():
        if G.nodes[ch]["energy"] <= 0:
            continue

        rx_energy = len(members) * calculate_reception_energy()
        da_energy = len(members) * calculate_aggregation_energy()
        safe_update_energy(ch, rx_energy + da_energy, G)

    # CH -> BS
    for ch in cluster_heads:
        if G.nodes[ch]["energy"] <= 0:
            continue

        d_ch_bs = euclidean_distance(positions[ch], base_station)
        tx_bs_energy = calculate_transmission_energy(d_ch_bs)
        safe_update_energy(ch, tx_bs_energy, G)


# =========================================================
# Main Protocol Function
# =========================================================
def cro_protocol(nodes, positions, transmission_range, G, model_save_path=None):
    """
    CRO-Ulite protocol for one simulation round.

    Parameters
    ----------
    nodes : list
        Node IDs
    positions : dict
        Node positions {node_id: (x, y)}
    transmission_range : float
        Kept for compatibility with project interface
    G : networkx.Graph
        Graph containing node energy and other attributes
    model_save_path : str or None
        Kept for compatibility with old interface; unused here

    Returns
    -------
    list
        Selected cluster head node IDs
    """
    del transmission_range
    del model_save_path

    alive_nodes = get_alive_nodes(nodes, G)
    if not alive_nodes:
        return []

    candidates = pca_kmeans_filter(
        alive_nodes=alive_nodes,
        positions=positions,
        G=G,
        base_station=BASE_STATION,
        ch_ratio=0.2,
        pca_components=2,
    )

    if not candidates:
        return []

    optimizer = CoralReefOptimizer(
        candidates=candidates,
        alive_nodes=alive_nodes,
        positions=positions,
        G=G,
        base_station=BASE_STATION,
        reef_size=20,
        n_iterations=30,
        fa=0.1,
        fd=0.1,
        mutation_rate=0.3,
        k_attempts=3,
    )

    cluster_heads = optimizer.optimize()
    if not cluster_heads:
        return []

    clusters = assign_clusters(alive_nodes, cluster_heads, positions)
    update_cluster_energy(clusters, cluster_heads, positions, G, BASE_STATION)

    return cluster_heads
