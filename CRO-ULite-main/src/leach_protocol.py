import random
import numpy as np

from src.energy_model import (
    calculate_energy_consumption,
    update_energy,
    apply_pca
)
from src.clustering import kmeans_cluster_nodes


# =========================
# Configuration
# =========================
BASE_STATION = (50, 175)   # same as your network setup
CH_RATIO = 0.2             # 20% of alive nodes become cluster heads
E_RX = 5e-8                # energy for receiving one aggregated unit
E_DA = 5e-9                # data aggregation energy per received member packet


def euclidean_distance(p1, p2):
    """Compute Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def safe_update_energy(node, energy_spent, G):
    """
    Update node energy and clamp it to zero if numerical drift occurs.
    """
    update_energy(node, energy_spent, G)
    if G.nodes[node]["energy"] < 0:
        G.nodes[node]["energy"] = 0


def get_alive_nodes(nodes, G):
    """Return only nodes with positive residual energy."""
    return [node for node in nodes if G.nodes[node]["energy"] > 0]


def select_random_cluster_heads(alive_nodes, ratio=CH_RATIO):
    """
    Randomly select cluster heads from alive nodes.
    """
    if not alive_nodes:
        return []

    num_ch = max(1, int(len(alive_nodes) * ratio))
    num_ch = min(num_ch, len(alive_nodes))
    return random.sample(alive_nodes, num_ch)


def assign_nodes_to_cluster_heads(alive_nodes, cluster_heads, positions):
    """
    Assign each non-CH alive node to its nearest cluster head.

    Returns:
        clusters: dict mapping CH -> list of member nodes
    """
    clusters = {ch: [] for ch in cluster_heads}

    for node in alive_nodes:
        if node in cluster_heads:
            continue

        closest_head = min(
            cluster_heads,
            key=lambda ch: euclidean_distance(positions[node], positions[ch])
        )
        clusters[closest_head].append(node)

    return clusters


def member_to_ch_energy(clusters, positions, transmission_range, G):
    """
    Energy spent by member nodes transmitting data to their assigned CH.
    """
    for ch, members in clusters.items():
        for node in members:
            distance = euclidean_distance(positions[node], positions[ch])
            energy_spent = calculate_energy_consumption(distance, transmission_range)
            safe_update_energy(node, energy_spent, G)


def ch_receive_and_aggregate_energy(clusters, G):
    """
    Energy spent by each CH receiving packets from members and aggregating them.
    """
    for ch, members in clusters.items():
        if G.nodes[ch]["energy"] <= 0:
            continue

        receive_cost = len(members) * E_RX
        aggregation_cost = len(members) * E_DA
        total_ch_processing_cost = receive_cost + aggregation_cost

        safe_update_energy(ch, total_ch_processing_cost, G)


def ch_to_bs_energy(cluster_heads, positions, transmission_range, G, base_station=BASE_STATION):
    """
    Energy spent by each cluster head transmitting aggregated data to the BS.
    """
    for ch in cluster_heads:
        if G.nodes[ch]["energy"] <= 0:
            continue

        distance_to_bs = euclidean_distance(positions[ch], base_station)
        energy_spent = calculate_energy_consumption(distance_to_bs, transmission_range)
        safe_update_energy(ch, energy_spent, G)


def leach_protocol(nodes, positions, transmission_range, G):
    """
    Simplified LEACH baseline.

    Steps:
    1. Select cluster heads randomly from alive nodes.
    2. Assign alive member nodes to nearest CH.
    3. Members transmit to CH.
    4. CH receives and aggregates data.
    5. CH transmits aggregated data to BS.

    Returns:
        cluster_heads: list of selected cluster heads
    """
    alive_nodes = get_alive_nodes(nodes, G)
    if not alive_nodes:
        return []

    cluster_heads = select_random_cluster_heads(alive_nodes, ratio=CH_RATIO)
    if not cluster_heads:
        return []

    clusters = assign_nodes_to_cluster_heads(alive_nodes, cluster_heads, positions)

    member_to_ch_energy(clusters, positions, transmission_range, G)
    ch_receive_and_aggregate_energy(clusters, G)
    ch_to_bs_energy(cluster_heads, positions, transmission_range, G, base_station=BASE_STATION)

    return cluster_heads


def leach_with_pca_kmeans(nodes, positions, transmission_range, G):
    """
    Modified LEACH using PCA + K-means for cluster formation and
    highest-energy node in each cluster as the cluster head.

    Steps:
    1. Build feature matrix from alive nodes only.
    2. Apply PCA.
    3. Cluster nodes using K-means.
    4. Select the highest-energy node in each cluster as CH.
    5. Members transmit to CH.
    6. CH receives and aggregates data.
    7. CH transmits aggregated data to BS.

    Returns:
        cluster_heads: list of selected cluster heads
    """
    alive_nodes = get_alive_nodes(nodes, G)
    if not alive_nodes:
        return []

    node_data = []
    active_nodes = []

    for node in alive_nodes:
        active_nodes.append(node)
        node_data.append([
            positions[node][0],
            positions[node][1],
            G.nodes[node]["energy"],
            G.nodes[node].get("signal_strength", 0)
        ])

    if not node_data:
        return []

    reduced_data = apply_pca(node_data)

    n_clusters = max(1, int(len(active_nodes) * CH_RATIO))
    n_clusters = min(n_clusters, len(active_nodes))

    _, labels = kmeans_cluster_nodes(reduced_data, n_clusters=n_clusters)

    clustered_nodes = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_nodes[label].append(active_nodes[idx])

    cluster_heads = []
    for cluster_id, cluster_members in clustered_nodes.items():
        if not cluster_members:
            continue

        ch = max(cluster_members, key=lambda node: G.nodes[node]["energy"])
        cluster_heads.append(ch)

    if not cluster_heads:
        return []

    clusters = {ch: [] for ch in cluster_heads}

    # Assign each alive non-CH node to nearest selected CH
    for node in active_nodes:
        if node in cluster_heads:
            continue

        closest_head = min(
            cluster_heads,
            key=lambda ch: euclidean_distance(positions[node], positions[ch])
        )
        clusters[closest_head].append(node)

    member_to_ch_energy(clusters, positions, transmission_range, G)
    ch_receive_and_aggregate_energy(clusters, G)
    ch_to_bs_energy(cluster_heads, positions, transmission_range, G, base_station=BASE_STATION)

    return cluster_heads
