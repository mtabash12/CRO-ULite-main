import random
import networkx as nx
import math

# Paper-aligned simulation settings
NUM_NODES = 100
FIELD_WIDTH = 100
FIELD_HEIGHT = 100
INITIAL_ENERGY = 0.5
BS_LOCATION = (50, 175)
TRANSMISSION_RANGE = 100
RANDOM_SEED = 42


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def create_wsn(
    num_nodes=NUM_NODES,
    field_width=FIELD_WIDTH,
    field_height=FIELD_HEIGHT,
    initial_energy=INITIAL_ENERGY,
    bs_location=BS_LOCATION,
    transmission_range=TRANSMISSION_RANGE,
    seed=RANDOM_SEED,
):
    random.seed(seed)

    G = nx.Graph()

    # Add sensor nodes
    for i in range(num_nodes):
        x = random.uniform(0, field_width)
        y = random.uniform(0, field_height)

        dist_to_bs = euclidean_distance((x, y), bs_location)

        G.add_node(
            i,
            pos=(x, y),
            energy=initial_energy,
            alive=True,
            dist_to_bs=dist_to_bs,
            is_CH=False,
            cluster=None,
        )

    # Optional: add connectivity edges based on transmission range
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pos_i = G.nodes[nodes[i]]["pos"]
            pos_j = G.nodes[nodes[j]]["pos"]
            d = euclidean_distance(pos_i, pos_j)

            if d <= transmission_range:
                G.add_edge(nodes[i], nodes[j], distance=d)

    return G, bs_location
