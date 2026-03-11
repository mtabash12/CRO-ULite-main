import random
import numpy as np

from src.energy_model import (
    calculate_transmission_energy,
    calculate_reception_energy,
    calculate_aggregation_energy,
    update_energy,
    apply_pca
)
from src.clustering import kmeans_cluster_nodes
from src.network_setup import BASE_STATION


CH_RATIO = 0.2


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def safe_update_energy(node, energy_spent, G):
    update_energy(node, energy_spent, G)
    if G.nodes[node]["energy"] < 0:
        G.nodes[node]["energy"] = 0


def get_alive_nodes(nodes, G):
    return [node for node in nodes if G.nodes[node]["energy"] > 0]


def select_random_cluster_heads(alive_nodes, ratio=CH_RATIO):
    if not alive_nodes:
        return []

    num_ch = max(1, int(len(alive_nodes) * ratio))
    num_ch = min(num_ch, len(alive_nodes))
    return random.sample(alive_nodes, num_ch)


def assign_nodes_to_cluster_heads(alive_nodes, cluster_heads, positions):
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


def member_to_ch_energy(clusters, positions, G):
    for ch, members in clusters.items():
        for node in members:
            distance = euclidean_distance(positions[node], positions[ch])
            energy_spent = calculate_transmission_energy(distance)
            safe_update_energy(node, energy_spent, G)


def ch_receive_and_aggregate_energy(clusters, G):
    for ch, members in clusters.items():
        if G.nodes[ch]["energy"] <= 0:
            continue

        receive_cost = len(members) * calculate_reception_energy()
        aggregation_cost = len(members) * calculate_aggregation_energy()
        safe_update_energy(ch, receive_cost + aggregation_cost, G)


def ch_to_bs_energy(cluster_heads, positions, G, base_station=BASE_STATION):
    for ch in cluster_heads:
        if G.nodes[ch]["energy"] <= 0:
            continue

        distance_to_bs = euclidean_distance(positions[ch], base_station)
        energy_spent = calculate_transmission_energy(distance_to_bs)
        safe_update_energy(ch, energy_spent, G)


def leach_protocol(nodes, positions, transmission_range, G):
    del transmission_range

    alive_nodes = get_alive_nodes(nodes, G)
    if not alive_nodes:
        return []

    cluster_heads = select_random_cluster_heads(alive_nodes, ratio=CH_RATIO)
    if not cluster_heads:
        return []

    clusters = assign_nodes_to_cluster_heads(alive_nodes, cluster_heads, positions)

    member_to_ch_energy(clusters, positions, G)
    ch_receive_and_aggregate_energy(clusters, G)
    ch_to_bs_energy(cluster_heads, positions, G, base_station=BASE_STATION)

    return cluster_heads


def leach_with_pca_kmeans(nodes, positions, transmission_range, G):
    del transmission_range

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
    for _, cluster_members in clustered_nodes.items():
        if cluster_members:
            ch = max(cluster_members, key=lambda node: G.nodes[node]["energy"])
            cluster_heads.append(ch)

    if not cluster_heads:
        return []

    clusters = assign_nodes_to_cluster_heads(active_nodes, cluster_heads, positions)

    member_to_ch_energy(clusters, positions, G)
    ch_receive_and_aggregate_energy(clusters, G)
    ch_to_bs_energy(cluster_heads, positions, G, base_station=BASE_STATION)

    return cluster_heads
