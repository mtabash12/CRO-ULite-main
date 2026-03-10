# CRO-Ulite: Energy-Efficient Cluster Head Selection for Wireless Sensor Networks

CRO-Ulite is a lightweight hybrid framework designed to improve **Cluster Head (CH) selection** in Wireless Sensor Networks (WSNs).  
The framework combines **dimensionality reduction, clustering, and metaheuristic optimization** to enhance network lifetime and reduce energy consumption.

The proposed method integrates:

- **Principal Component Analysis (PCA)** for feature dimensionality reduction
- **K-Means clustering** for candidate CH filtering
- **Coral Reef Optimization (CRO)** for optimal cluster head selection

This hybrid pipeline improves the energy efficiency of clustering-based routing protocols while maintaining low computational overhead suitable for **resource-constrained WSN environments**.

---

# Methodology

The CRO-Ulite framework follows a **three-stage pipeline**:

1. **Feature Extraction**
   - Node attributes:  
   \[ x_i, y_i, E_i, d_{BS} \]

2. **PCA Dimensionality Reduction**
   - Reduces feature space
   - Removes correlation and noise

3. **K-Means Candidate Filtering**
   - Groups nodes into clusters
   - Selects high-energy nodes as CH candidates

4. **Coral Reef Optimization**
   - Searches for the optimal subset of cluster heads
   - Optimizes energy and communication distance

---

# Simulation Setup

The simulation environment is configured as follows:

| Parameter         | Value |
| Number of nodes   | 100 |
| Network field     | 100 × 100 m² |
| Base station      | (50, 175) |
| Initial energy    | 0.5 J |
| Packet size       | 4000 bits |
| Simulation rounds | 1000 |
| Random seed       | 42 |

The first-order radio energy model is used to simulate communication energy consumption.

---

# Compared Protocols

The performance of CRO-Ulite is evaluated against:

- **LEACH**
- **LEACH + PCA + K-Means**

---

# Key Results

The proposed framework significantly improves network lifetime and energy efficiency.

| Metric                 | LEACH   | LEACH+PCA+KMeans | CRO-Ulite   |
| FND (First Node Death) | 698     | 753              | **961**     |
| Avg Alive Nodes        | 90.7    | 93.3             | **99.1**    |
| Energy @R100           | 5.36 J  | 5.17 J           | **4.90 J**  |
| Energy @R500           | 26.85 J | 26.03 J          | **25.17 J** |


 




**Improvements achieved by CRO-Ulite:**

- **37.7% improvement over LEACH**
- **27.6% improvement over LEACH + PCA + K-Means**

---

# Repository Structure


CRO-Ulite-main
│
├── src/ Simulation and protocol implementation
├── models/ Machine learning components
├── data/ Generated network data
├── outputs/ Simulation results and plots
└── README.md Project documentation


---

# Running the Simulation

### Install dependencies


pip install networkx numpy matplotlib scikit-learn


### Run the simulation


python src/main.py


The simulation will generate plots and performance results in the **outputs** folder.

---

# Reproducibility

All experiments in the paper were conducted using:


Random seed = 42


This ensures reproducible simulation results.

---

# Citation

If you use this repository in your research, please cite:


Author,
"CRO-Ulite: Energy-Efficient Cluster Head Selection for Wireless Sensor Networks",
2025.


---

# License

This project is intended for **academic and research purposes**.
