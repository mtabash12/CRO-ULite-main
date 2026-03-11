# Standard library imports
import os
import copy
import traceback
from functools import partial

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
    plot_protocol_comparison
)
from src.edge_deployment import deploy_model_on_rpi


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

NETWORK_TOPOLOGY_PATH = os.path.join(OUTPUTS_DIR, "network_topology.png")
INITIAL_STATE_CSV_PATH = os.path.join(DATA_DIR, "initial_network_state.csv")
PROTOCOL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "protocol_comparison.png")
MODEL_PATH = os.path.join(MODEL_DIR, "cro_model.h5")
QUANTIZED_MODEL_PATH = os.path.join(MODEL_DIR, "cro_model_quantized.tflite")


def ensure_directories():
    """Create required directories if they do not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def initialize_and_save_network():
    """
    Initialize the network, save the initial node state to CSV,
    and plot the topology.
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
            "signal_strength": G.nodes[node].get("signal_strength", 0)
        })

    print("Plotting network topology...")
    plot_network_topology(positions, save_path=NETWORK_TOPOLOGY_PATH)

    results_df = pd.DataFrame(node_data)
    results_df.to_csv(INITIAL_STATE_CSV_PATH, index=False)
    print(f"Initial network state saved to: {INITIAL_STATE_CSV_PATH}")

    return G, positions


def run_protocol_comparison(G, positions):
    """
    Run and compare the selected routing protocols.
    
    Note:
    For strict scientific fairness, each protocol should run on an
    independent copy of the same initial network state.
    """

    print("Comparing protocols...")
    print("Protocols: LEACH + PCA/K-means, CRO-ULite, LEACH")

    cro_protocol_with_path = partial(cro_protocol, model_save_path=MODEL_PATH)

    protocols = [
        leach_with_pca_kmeans,
        cro_protocol_with_path,
        leach_protocol
    ]

    protocol_names = [
        "LEACH with PCA and K-means",
        "CRO-ULite (Our Model)",
        "LEACH without PCA and K-means"
    ]

    # If compare_protocols does NOT internally clone G for each protocol,
    # it should be updated there for fair benchmarking.
    comparison_results = compare_protocols(
        list(range(NUM_NODES)),
        protocols,
        protocol_names,
        max_rounds=200,
        G=G,
        positions=positions,
        transmission_range=TRANSMISSION_RANGE
    )

    return comparison_results


def plot_results(comparison_results):
    """Plot overall and per-protocol benchmark results."""
    print("Plotting comparison results...")

    plot_protocol_comparison(
        comparison_results,
        save_path=PROTOCOL_COMPARISON_PATH
    )

    protocol_results = comparison_results.get("protocol_results", {})
    for protocol_name, results in protocol_results.items():
        safe_filename = protocol_name.replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(OUTPUTS_DIR, f"{safe_filename}_results.png")

        plot_benchmark_results(
            results,
            save_path=output_path
        )

    print(f"All plots saved to: {OUTPUTS_DIR}")


def deploy_model():
    """
    Convert/deploy the trained CRO model for edge use.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found, skipping deployment: {MODEL_PATH}")
            return

        print("Deploying model for edge device...")
        deploy_model_on_rpi(MODEL_PATH, QUANTIZED_MODEL_PATH)
        print(f"Model deployment completed: {QUANTIZED_MODEL_PATH}")

    except Exception as e:
        print(f"Model deployment failed: {e}")
        traceback.print_exc()


def main():
    """Main execution pipeline."""
    try:
        ensure_directories()

        # Step 1: Initialize network
        G, positions = initialize_and_save_network()

        # Step 2: Compare protocols
        comparison_results = run_protocol_comparison(G, positions)

        # Step 3: Plot results
        plot_results(comparison_results)

        # Step 4: Deploy trained model if available
        deploy_model()

        print("Simulation workflow completed successfully.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
