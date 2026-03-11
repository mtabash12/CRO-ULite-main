# Standard library imports
import os
import traceback

# Third-party imports
import pandas as pd

# Local imports
from src.network_setup import initialize_network, NUM_NODES, TRANSMISSION_RANGE
from src.leach_protocol import leach_protocol, leach_with_pca_kmeans
from src.cro_protocol import cro_protocol
from src.benchmarking import compare_protocols
from src.visualization import (
    plot_network_topology,
    plot_benchmark_results,
    plot_protocol_comparison,
)


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

NETWORK_TOPOLOGY_PATH = os.path.join(OUTPUTS_DIR, "network_topology.png")
INITIAL_STATE_CSV_PATH = os.path.join(DATA_DIR, "initial_network_state.csv")
PROTOCOL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "protocol_comparison.png")


def ensure_directories():
    """Create required directories if they do not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)


def initialize_and_save_network():
    """
    Initialize the network, save the initial node state to CSV,
    and plot the topology.

    Returns
    -------
    tuple
        G : networkx.Graph
            Initialized WSN graph.
        positions : dict
            Dictionary mapping node_id -> (x, y).
    """
    print("Initializing network...")
    G, positions, _ = initialize_network()

    node_data = []
    for node in range(NUM_NODES):
        node_data.append({
            "node_id": node,
            "x": positions[node][0],
            "y": positions[node][1],
            "energy": G.nodes[node]["energy"],
            "signal_strength": G.nodes[node].get("signal_strength", 0.0),
            "dist_to_bs": G.nodes[node].get("dist_to_bs", None),
        })

    results_df = pd.DataFrame(node_data)
    results_df.to_csv(INITIAL_STATE_CSV_PATH, index=False)
    print(f"Initial network state saved to: {INITIAL_STATE_CSV_PATH}")

    print("Plotting network topology...")
    plot_network_topology(positions, save_path=NETWORK_TOPOLOGY_PATH)
    print(f"Network topology plot saved to: {NETWORK_TOPOLOGY_PATH}")

    return G, positions


def run_protocol_comparison(G, positions):
    """
    Run and compare the selected routing protocols using
    the same initial network state.
    """
    print("Comparing protocols...")
    print("Protocols: LEACH-PCA-KMeans, CRO-ULite, Simplified LEACH")

    protocols = [
        leach_with_pca_kmeans,
        cro_protocol,
        leach_protocol,
    ]

    protocol_names = [
        "LEACH-PCA-KMeans",
        "CRO-ULite",
        "Simplified LEACH",
    ]

    comparison_results = compare_protocols(
        nodes=list(range(NUM_NODES)),
        protocols=protocols,
        protocol_names=protocol_names,
        max_rounds=200,
        G=G,
        positions=positions,
        transmission_range=TRANSMISSION_RANGE,
    )

    return comparison_results


def save_protocol_results(comparison_results):
    """
    Save per-protocol simulation results to CSV files.
    """
    protocol_results = comparison_results.get("protocol_results", {})

    for protocol_name, results in protocol_results.items():
        safe_filename = protocol_name.replace(" ", "_").replace("/", "_").lower()
        csv_path = os.path.join(DATA_DIR, f"{safe_filename}_results.csv")

        result_df = pd.DataFrame({
            "round": results.get("rounds", []),
            "alive_nodes": results.get("alive_nodes", []),
            "total_energy": results.get("total_energy", []),
        })

        result_df.to_csv(csv_path, index=False)
        print(f"{protocol_name} results saved to: {csv_path}")


def plot_results(comparison_results):
    """
    Plot overall comparison results and individual protocol results.
    """
    print("Plotting comparison results...")

    plot_protocol_comparison(
        comparison_results,
        save_path=PROTOCOL_COMPARISON_PATH,
    )
    print(f"Protocol comparison plot saved to: {PROTOCOL_COMPARISON_PATH}")

    protocol_results = comparison_results.get("protocol_results", {})
    for protocol_name, results in protocol_results.items():
        safe_filename = protocol_name.replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(OUTPUTS_DIR, f"{safe_filename}_results.png")

        plot_benchmark_results(
            results,
            save_path=output_path,
        )
        print(f"{protocol_name} plot saved to: {output_path}")


def print_summary(comparison_results):
    """
    Print a concise summary of protocol performance.
    """
    print("\nSimulation Summary")
    print("=" * 50)

    protocol_results = comparison_results.get("protocol_results", {})
    for protocol_name, results in protocol_results.items():
        fnd = results.get("fnd", "N/A")
        hnd = results.get("hnd", "N/A")
        lnd = results.get("lnd", "N/A")

        print(f"{protocol_name}:")
        print(f"  FND = {fnd}")
        print(f"  HND = {hnd}")
        print(f"  LND = {lnd}")
        print("-" * 50)


def main():
    """
    Main execution pipeline.
    """
    try:
        ensure_directories()

        # Step 1: Initialize network
        G, positions = initialize_and_save_network()

        # Step 2: Compare protocols
        comparison_results = run_protocol_comparison(G, positions)

        # Step 3: Save numeric results
        save_protocol_results(comparison_results)

        # Step 4: Plot results
        plot_results(comparison_results)

        # Step 5: Print summary
        print_summary(comparison_results)

        print("Simulation workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
