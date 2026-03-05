"""
cro_protocol.py
================
CRO-Ulite: Coral Reef Optimization for Cluster Head Selection in WSNs
----------------------------------------------------------------------
Pipeline:
    Stage 1 — PCA + K-means  : reduces node features and filters candidates
    Stage 2 — CRO Optimizer  : selects optimal CH subset from candidates
    Stage 3 — Cluster assign : assigns every member node to its nearest CH

Usage (standalone):
    from cro_protocol import CROUlite
    cro = CROUlite(nodes, base_station=(50, 175))
    cluster_heads = cro.select_cluster_heads()
"""

import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — FITNESS FUNCTION
# ═══════════════════════════════════════════════════════════════
def compute_fitness(candidate_CHs, all_alive_nodes, base_station, max_dist=141.4):
    """
    Fitness function for evaluating a set of Cluster Heads.

    A higher fitness value means a better CH selection.

    Formula:
        fitness = w1 * avg_CH_energy
                - w2 * avg_member_to_CH_distance   (normalized)
                - w3 * avg_CH_to_BS_distance        (normalized)

    Weights (w1=0.4, w2=0.3, w3=0.3) balance energy vs. distance.

    Parameters
    ----------
    candidate_CHs   : list of node objects selected as CHs
    all_alive_nodes : list of all alive node objects in network
    base_station    : tuple (x, y) of the base station coordinates
    max_dist        : normalization factor for distances (default = diagonal of 100x100 area)

    Returns
    -------
    float : fitness score (higher = better)
    """
    if not candidate_CHs:
        return -np.inf

    # ── Metric 1: Average residual energy of selected CHs (normalized 0→1)
    avg_ch_energy = np.mean([n.energy for n in candidate_CHs]) / max(
        n.energy for n in all_alive_nodes
    )

    # ── Metric 2: Average distance from every member node to its nearest CH
    member_nodes = [n for n in all_alive_nodes if n not in candidate_CHs]
    if member_nodes:
        distances_to_ch = [
            min(
                _euclidean(m.x, m.y, ch.x, ch.y) for ch in candidate_CHs
            )
            for m in member_nodes
        ]
        avg_member_dist = np.mean(distances_to_ch) / max_dist
    else:
        avg_member_dist = 0.0

    # ── Metric 3: Average distance from each CH to the Base Station
    bx, by = base_station
    avg_ch_bs_dist = np.mean(
        [_euclidean(ch.x, ch.y, bx, by) for ch in candidate_CHs]
    ) / (max_dist * 2)   # BS can be outside the field

    # ── Weighted fitness
    fitness = (0.4 * avg_ch_energy
             - 0.3 * avg_member_dist
             - 0.3 * avg_ch_bs_dist)
    return fitness


def _euclidean(x1, y1, x2, y2):
    """Euclidean distance between two 2-D points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — STAGE 1: PCA + K-MEANS CANDIDATE FILTER
# ═══════════════════════════════════════════════════════════════
def pca_kmeans_filter(alive_nodes, n_clusters=5, pca_components=2):
    """
    Stage 1 of CRO-Ulite pipeline.

    Steps:
        1. Build feature matrix  [x, y, energy, dist_to_BS]
        2. Normalize features with StandardScaler
        3. Apply PCA → reduce to `pca_components` dimensions
        4. Apply K-Means → group nodes into `n_clusters` clusters
        5. From each cluster select the node with highest residual energy
           as the CH candidate

    Why PCA first?
        - Removes correlated features and noise
        - Reduces computational load for K-Means
        - Helps avoid the "curse of dimensionality"

    Parameters
    ----------
    alive_nodes    : list of alive node objects
    n_clusters     : number of clusters (= number of CH candidates returned)
    pca_components : number of principal components to keep

    Returns
    -------
    list of node objects — one best candidate per cluster
    """
    n = len(alive_nodes)
    if n <= n_clusters:
        # Too few nodes — return all as candidates
        return alive_nodes

    # Step 1: Feature matrix
    # Each row = [x_coord, y_coord, residual_energy, distance_to_BS]
    # (distance_to_BS is a proxy for communication cost)
    bs_x, bs_y = 50, 175   # default BS position — overridden by CROUlite class
    features = np.array([
        [
            node.x,
            node.y,
            node.energy,
            _euclidean(node.x, node.y, bs_x, bs_y)
        ]
        for node in alive_nodes
    ], dtype=float)

    # Step 2: Normalize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 3: PCA
    n_comp = min(pca_components, features_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    features_pca = pca.fit_transform(features_scaled)

    # Step 4: K-Means
    k = min(n_clusters, n)
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(features_pca)

    # Step 5: Best candidate per cluster (highest energy)
    candidates = []
    for cluster_id in range(k):
        cluster_members = [
            alive_nodes[i] for i, lbl in enumerate(labels) if lbl == cluster_id
        ]
        if cluster_members:
            best = max(cluster_members, key=lambda nd: nd.energy)
            candidates.append(best)

    return candidates


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — STAGE 2: CORAL REEF OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
class CoralReefOptimizer:
    """
    Coral Reef Optimization (CRO) for selecting the best CH subset.

    Biological analogy:
        - Reef    → population of solutions (subsets of CH candidates)
        - Coral   → one solution = one subset of candidate nodes as CHs
        - Larva   → new solution produced by reproduction
        - Fitness → quality of a CH subset (see compute_fitness)

    Operators implemented:
        1. Sexual reproduction (broadcast spawning) — crossover two corals
        2. Asexual reproduction (budding)           — clone + mutate one coral
        3. Larva settlement                         — accept larva if better
        4. Depredation                              — remove worst corals

    Parameters
    ----------
    candidates     : list of candidate nodes (output of Stage 1)
    alive_nodes    : all alive nodes in the network
    base_station   : (x, y) tuple
    reef_size      : number of coral solutions in the reef   (default 20)
    n_iterations   : number of CRO iterations per round     (default 50)
    fa             : fraction for asexual reproduction      (default 0.1)
    fd             : fraction for depredation               (default 0.1)
    mutation_rate  : probability of mutating a larva        (default 0.3)
    k_attempts     : settlement attempts per larva          (default 3)
    """

    def __init__(
        self,
        candidates,
        alive_nodes,
        base_station=(50, 175),
        reef_size=20,
        n_iterations=50,
        fa=0.1,
        fd=0.1,
        mutation_rate=0.3,
        k_attempts=3,
    ):
        self.candidates    = candidates
        self.alive_nodes   = alive_nodes
        self.base_station  = base_station
        self.reef_size     = min(reef_size, max(1, 2 ** len(candidates)))
        self.n_iterations  = n_iterations
        self.fa            = fa
        self.fd            = fd
        self.mutation_rate = mutation_rate
        self.k_attempts    = k_attempts

    # ── Helpers ───────────────────────────────────────────────
    def _fitness(self, coral):
        return compute_fitness(coral, self.alive_nodes, self.base_station)

    def _random_coral(self):
        """Create a random coral: random subset of candidates."""
        k = random.randint(1, max(1, len(self.candidates) // 2))
        return random.sample(self.candidates, min(k, len(self.candidates)))

    # ── Reproduction operators ─────────────────────────────────
    def _broadcast_spawning(self, coral_a, coral_b):
        """
        Sexual reproduction: combine two parent corals.
        Takes the first half of coral_a and second half of coral_b.
        """
        combined = list(set(coral_a + coral_b))
        random.shuffle(combined)
        mid   = max(1, len(combined) // 2)
        larva = combined[:mid]
        return larva if larva else [random.choice(self.candidates)]

    def _budding(self, coral):
        """
        Asexual reproduction: clone a coral and add one random candidate.
        """
        bud = coral.copy()
        extra = random.choice(self.candidates)
        if extra not in bud:
            bud.append(extra)
        return list(set(bud))

    def _mutate(self, coral):
        """
        Replace one random element in the coral with another candidate.
        """
        if not coral:
            return coral
        mutated = coral.copy()
        idx = random.randrange(len(mutated))
        mutated[idx] = random.choice(self.candidates)
        return list(set(mutated))

    # ── Larva settlement ───────────────────────────────────────
    def _settle(self, reef, larva):
        """
        Try to settle a larva onto the reef.
        The larva replaces a random reef position if its fitness is better.
        Attempts `k_attempts` times before giving up.
        """
        larva_fitness = self._fitness(larva)
        for _ in range(self.k_attempts):
            target_idx = random.randrange(len(reef))
            if larva_fitness > self._fitness(reef[target_idx]):
                reef[target_idx] = larva
                return True
        return False

    # ── Main CRO loop ──────────────────────────────────────────
    def optimize(self):
        """
        Run the CRO algorithm and return the best CH subset found.

        Returns
        -------
        list of node objects — the optimal Cluster Heads
        """
        if not self.candidates:
            return []
        if len(self.candidates) == 1:
            return self.candidates

        # ── Initialize reef with random corals
        reef = [self._random_coral() for _ in range(self.reef_size)]

        best_coral   = max(reef, key=self._fitness)
        best_fitness = self._fitness(best_coral)

        # ── Main iteration loop
        for iteration in range(self.n_iterations):
            larvae = []

            # 1. Broadcast spawning (sexual reproduction)
            #    Pair up random corals and produce larvae
            random.shuffle(reef)
            for i in range(0, len(reef) - 1, 2):
                larva = self._broadcast_spawning(reef[i], reef[i + 1])
                # Optional mutation
                if random.random() < self.mutation_rate:
                    larva = self._mutate(larva)
                larvae.append(larva)

            # 2. Budding (asexual reproduction)
            #    Best coral in reef produces a bud
            n_buds = max(1, int(self.fa * self.reef_size))
            top_corals = sorted(reef, key=self._fitness, reverse=True)[:n_buds]
            for coral in top_corals:
                bud = self._budding(coral)
                if random.random() < self.mutation_rate:
                    bud = self._mutate(bud)
                larvae.append(bud)

            # 3. Larva settlement
            for larva in larvae:
                if larva:
                    self._settle(reef, larva)

            # 4. Depredation: remove the worst corals (keep reef healthy)
            n_remove = max(1, int(self.fd * self.reef_size))
            reef.sort(key=self._fitness, reverse=True)
            reef = reef[:max(1, len(reef) - n_remove)]

            # 5. Ensure reef size stays constant
            while len(reef) < self.reef_size:
                reef.append(self._random_coral())

            # 6. Track global best
            current_best = max(reef, key=self._fitness)
            current_fitness = self._fitness(current_best)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_coral   = current_best

        return best_coral if best_coral else self.candidates[:1]


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — STAGE 3: CLUSTER ASSIGNMENT
# ═══════════════════════════════════════════════════════════════
def assign_clusters(alive_nodes, cluster_heads):
    """
    Assign every non-CH node to its nearest Cluster Head.

    Parameters
    ----------
    alive_nodes   : list of all alive node objects
    cluster_heads : list of nodes selected as CHs

    Returns
    -------
    dict : { ch_node : [list of member nodes] }
    """
    clusters = {ch: [] for ch in cluster_heads}

    for node in alive_nodes:
        if node in cluster_heads:
            continue  # CH nodes don't join as members
        nearest_ch = min(
            cluster_heads,
            key=lambda ch: _euclidean(node.x, node.y, ch.x, ch.y)
        )
        clusters[nearest_ch].append(node)

    return clusters


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — MAIN CLASS: CROUlite (Full Pipeline)
# ═══════════════════════════════════════════════════════════════
class CROUlite:
    """
    CRO-Ulite: Full 3-stage pipeline for CH selection.

    Stage 1 → PCA + K-Means  : candidate filtering
    Stage 2 → CRO            : optimal CH selection
    Stage 3 → Assignment     : cluster member assignment

    Parameters
    ----------
    nodes          : list of all node objects in the network
    base_station   : tuple (x, y) — coordinates of the base station
    n_clusters     : number of K-Means clusters / CH candidates (default 5)
    pca_components : PCA output dimensions (default 2)
    reef_size      : CRO reef size (default 20)
    cro_iterations : CRO iterations per round (default 50)

    Example
    -------
    >>> cro = CROUlite(nodes, base_station=(50, 175))
    >>> CHs = cro.select_cluster_heads()
    >>> clusters = cro.get_clusters()
    """

    def __init__(
        self,
        nodes,
        base_station=(50, 175),
        n_clusters=5,
        pca_components=2,
        reef_size=20,
        cro_iterations=50,
    ):
        self.nodes         = nodes
        self.base_station  = base_station
        self.n_clusters    = n_clusters
        self.pca_components = pca_components
        self.reef_size     = reef_size
        self.cro_iterations = cro_iterations

        self._cluster_heads = []
        self._clusters      = {}

    def select_cluster_heads(self):
        """
        Run the full CRO-Ulite pipeline for one round.

        Returns
        -------
        list of node objects selected as Cluster Heads
        """
        alive = [n for n in self.nodes if n.is_alive]
        if not alive:
            return []

        # ── Stage 1: PCA + K-Means candidate filtering
        candidates = pca_kmeans_filter(
            alive_nodes    = alive,
            n_clusters     = self.n_clusters,
            pca_components = self.pca_components,
        )

        # ── Stage 2: CRO optimization
        optimizer = CoralReefOptimizer(
            candidates    = candidates,
            alive_nodes   = alive,
            base_station  = self.base_station,
            reef_size     = self.reef_size,
            n_iterations  = self.cro_iterations,
        )
        self._cluster_heads = optimizer.optimize()

        # Mark CH status on node objects
        for node in alive:
            node.is_CH   = (node in self._cluster_heads)
            node.CH_node = None

        # ── Stage 3: Assign members to CHs
        self._clusters = assign_clusters(alive, self._cluster_heads)

        for ch, members in self._clusters.items():
            for m in members:
                m.CH_node = ch

        return self._cluster_heads

    def get_clusters(self):
        """
        Return the cluster dictionary from the last round.

        Returns
        -------
        dict : { ch_node : [member_nodes] }
        """
        return self._clusters

    def get_stats(self):
        """
        Return a summary dict of the last round's CH selection.

        Returns
        -------
        dict with keys: num_CHs, avg_ch_energy, avg_cluster_size
        """
        if not self._cluster_heads:
            return {}
        return {
            "num_CHs"          : len(self._cluster_heads),
            "avg_ch_energy"    : round(np.mean([ch.energy for ch in self._cluster_heads]), 6),
            "avg_cluster_size" : round(
                np.mean([len(m) for m in self._clusters.values()]), 2
            ) if self._clusters else 0,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — QUICK SELF-TEST (run this file directly to test)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Minimal Node class for standalone testing
    class _TestNode:
        def __init__(self, nid, x, y, energy=0.5):
            self.id       = nid
            self.x        = x
            self.y        = y
            self.energy   = energy
            self.is_alive = True
            self.is_CH    = False
            self.CH_node  = None

    # ── Create 100 random nodes
    np.random.seed(42)
    test_nodes = [
        _TestNode(i, np.random.uniform(0, 100), np.random.uniform(0, 100))
        for i in range(100)
    ]

    # ── Run CRO-Ulite
    print("=" * 50)
    print("CRO-Ulite Self-Test")
    print("=" * 50)

    cro = CROUlite(test_nodes, base_station=(50, 175))
    CHs = cro.select_cluster_heads()

    print(f"✅ Cluster Heads selected : {len(CHs)}")
    for ch in CHs:
        print(f"   Node {ch.id:>3} | pos=({ch.x:.1f}, {ch.y:.1f}) | energy={ch.energy:.4f}")

    stats = cro.get_stats()
    print(f"\n📊 Stats: {stats}")

    clusters = cro.get_clusters()
    print(f"\n📦 Cluster sizes:")
    for ch, members in clusters.items():
        print(f"   CH {ch.id:>3} → {len(members)} members")

    print("\n✅ cro_protocol.py works correctly!")