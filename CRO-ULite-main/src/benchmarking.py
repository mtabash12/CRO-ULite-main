import copy
import numpy as np


def simulate_network_lifetime(
    nodes,
    protocol,
    max_rounds=100,
    G=None,
    positions=None,
    transmission_range=100
):
    """
    Simulate the network over multiple rounds using the specified protocol.

    Returns:
        dict with:
        - rounds
        - alive_nodes
        - total_energy
        - fnd: first node death round
        - hnd: half node death round
        - lnd: last node death round
    """
    results = {
        "rounds": [],
        "alive_nodes": [],
        "total_energy": [],
        "fnd": None,
        "hnd": None,
        "lnd": None
    }

    total_nodes = len(nodes)
    half_nodes_threshold = total_nodes / 2

    for round_num in range(max_rounds):
        # Run protocol first
        protocol(nodes, positions, transmission_range, G)

        # Clamp negative energies if needed
        for node in nodes:
            if G.nodes[node]["energy"] < 0:
                G.nodes[node]["energy"] = 0

        # Record state after protocol execution
        alive_nodes = [node for node in nodes if G.nodes[node]["energy"] > 0]
        dead_nodes = total_nodes - len(alive_nodes)
        total_energy = sum(G.nodes[node]["energy"] for node in nodes)

        results["rounds"].append(round_num + 1)
        results["alive_nodes"].append(len(alive_nodes))
        results["total_energy"].append(total_energy)

        # First Node Dies
        if results["fnd"] is None and dead_nodes >= 1:
            results["fnd"] = round_num + 1

        # Half Nodes Die
        if results["hnd"] is None and dead_nodes >= half_nodes_threshold:
            results["hnd"] = round_num + 1

        # Last Node Dies
        if dead_nodes == total_nodes:
            results["lnd"] = round_num + 1
            break

    # If no death event occurred within max_rounds
    if results["fnd"] is None:
        results["fnd"] = max_rounds
    if results["hnd"] is None:
        results["hnd"] = max_rounds
    if results["lnd"] is None:
        results["lnd"] = max_rounds

    return results


def compare_protocols(
    nodes,
    protocols,
    protocol_names,
    max_rounds=100,
    G=None,
    positions=None,
    transmission_range=100
):
    """
    Compare multiple protocols using the same initial network state.
    """
    comparison_results = {
        "protocol_results": {}
    }

    for protocol, name in zip(protocols, protocol_names):
        print(f"Simulating {name}...")

        G_copy = copy.deepcopy(G)

        results = simulate_network_lifetime(
            nodes=nodes,
            protocol=protocol,
            max_rounds=max_rounds,
            G=G_copy,
            positions=positions,
            transmission_range=transmission_range
        )

        comparison_results["protocol_results"][name] = results

        print(
            f"{name}: "
            f"FND={results['fnd']}, "
            f"HND={results['hnd']}, "
            f"LND={results['lnd']}"
        )

    return comparison_results
