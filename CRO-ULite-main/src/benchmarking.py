import copy


def simulate_network_lifetime(
    nodes,
    protocol,
    max_rounds=100,
    G=None,
    positions=None,
    transmission_range=100,
):
    """
    Simulate network lifetime for a single protocol.

    Parameters
    ----------
    nodes : list
        List of node IDs.
    protocol : callable
        Protocol function with signature:
        protocol(nodes, positions, transmission_range, G)
    max_rounds : int, optional
        Maximum number of simulation rounds.
    G : networkx.Graph
        Network graph containing node states.
    positions : dict
        Dictionary mapping node_id -> (x, y).
    transmission_range : float, optional
        Transmission range parameter kept for interface compatibility.

    Returns
    -------
    dict
        Simulation results containing:
        - rounds
        - alive_nodes
        - total_energy
        - fnd
        - hnd
        - lnd
    """
    if G is None:
        raise ValueError("G must not be None.")
    if positions is None:
        raise ValueError("positions must not be None.")
    if not nodes:
        raise ValueError("nodes list must not be empty.")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be greater than 0.")

    results = {
        "rounds": [],
        "alive_nodes": [],
        "total_energy": [],
        "fnd": None,
        "hnd": None,
        "lnd": None,
    }

    total_nodes = len(nodes)
    half_nodes_threshold = total_nodes / 2

    for round_num in range(1, max_rounds + 1):
        # Run one protocol round
        protocol(nodes, positions, transmission_range, G)

        # Clamp any negative numerical drift
        for node in nodes:
            if G.nodes[node]["energy"] < 0:
                G.nodes[node]["energy"] = 0

        alive_count = sum(1 for node in nodes if G.nodes[node]["energy"] > 0)
        dead_count = total_nodes - alive_count
        total_energy = sum(G.nodes[node]["energy"] for node in nodes)

        results["rounds"].append(round_num)
        results["alive_nodes"].append(alive_count)
        results["total_energy"].append(total_energy)

        if results["fnd"] is None and dead_count >= 1:
            results["fnd"] = round_num

        if results["hnd"] is None and dead_count >= half_nodes_threshold:
            results["hnd"] = round_num

        if dead_count == total_nodes:
            results["lnd"] = round_num
            break

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
    transmission_range=100,
):
    """
    Compare multiple protocols using the same initial network state.

    Parameters
    ----------
    nodes : list
        List of node IDs.
    protocols : list
        List of protocol functions.
    protocol_names : list
        List of protocol names corresponding to `protocols`.
    max_rounds : int, optional
        Maximum simulation rounds.
    G : networkx.Graph
        Initial network graph.
    positions : dict
        Dictionary mapping node_id -> (x, y).
    transmission_range : float, optional
        Transmission range parameter.

    Returns
    -------
    dict
        Comparison results with per-protocol simulation outputs.
    """
    if G is None:
        raise ValueError("G must not be None.")
    if positions is None:
        raise ValueError("positions must not be None.")
    if not nodes:
        raise ValueError("nodes list must not be empty.")
    if len(protocols) != len(protocol_names):
        raise ValueError("protocols and protocol_names must have the same length.")

    comparison_results = {
        "protocol_results": {}
    }

    for protocol, protocol_name in zip(protocols, protocol_names):
        print(f"Simulating {protocol_name}...")

        protocol_graph = copy.deepcopy(G)

        results = simulate_network_lifetime(
            nodes=nodes,
            protocol=protocol,
            max_rounds=max_rounds,
            G=protocol_graph,
            positions=positions,
            transmission_range=transmission_range,
        )

        comparison_results["protocol_results"][protocol_name] = results

        print(
            f"{protocol_name}: "
            f"FND={results['fnd']}, "
            f"HND={results['hnd']}, "
            f"LND={results['lnd']}"
        )

    return comparison_results
