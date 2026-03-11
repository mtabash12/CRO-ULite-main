import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# Radio Energy Model Parameters
# =========================
E_ELEC = 50e-9      # Energy dissipated per bit to run transmitter/receiver circuitry
E_FS   = 10e-12     # Free space amplifier energy
E_MP   = 0.0013e-12 # Multi-path amplifier energy
E_DA   = 5e-9       # Data aggregation energy per bit
PACKET_SIZE = 4000  # bits

# Threshold distance
D0 = np.sqrt(E_FS / E_MP)


def calculate_transmission_energy(distance, packet_size=PACKET_SIZE):
    """
    Calculate transmission energy based on the first-order radio model.
    """
    if distance < D0:
        return packet_size * (E_ELEC + E_FS * (distance ** 2))
    return packet_size * (E_ELEC + E_MP * (distance ** 4))


def calculate_reception_energy(packet_size=PACKET_SIZE):
    """
    Calculate reception energy.
    """
    return packet_size * E_ELEC


def calculate_aggregation_energy(packet_size=PACKET_SIZE):
    """
    Calculate data aggregation energy.
    """
    return packet_size * E_DA


def calculate_energy_consumption(distance, transmission_range=None, packet_size=PACKET_SIZE):
    """
    Backward-compatible wrapper for legacy code.
    Ignores transmission_range and uses the radio model based on distance.
    """
    return calculate_transmission_energy(distance, packet_size=packet_size)


def update_energy(node, energy_spent, G):
    """
    Subtract spent energy from the node and clamp to zero.
    """
    G.nodes[node]["energy"] -= energy_spent
    if G.nodes[node]["energy"] < 0:
        G.nodes[node]["energy"] = 0


def apply_pca(node_data, n_components=2):
    """
    Apply PCA after feature scaling.
    """
    data = np.array(node_data, dtype=float)

    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < n_components:
        return data

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data
