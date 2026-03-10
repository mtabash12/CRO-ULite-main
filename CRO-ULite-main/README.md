
 CRO-Ulite uses a simplified three-stage pipeline: PCA → K-Means → CRO. The experiments reported in the paper correspond to this final configuration.

## How to Install and Run the Project

1. **Clone the repository or unzip the project**:
    ```bash
    git clone <repository_url>
    cd wsn_project
    ```

2. **Install dependencies**:
    - Create a virtual environment (recommended):
      ```bash
      python3 -m venv venv
      source venv/bin/activate  # For Windows: venv\Scripts\activate
      ```
    - Install required dependencies:
      ```bash
      pip install -r requirements.txt
      ```

3. **Run the simulation**:
    - To run the project, simply execute:
      ```bash
      python3 src/main.py
      

5. **Results**:
    - After running the simulation, the network topology and benchmark results will be saved in the `outputs` folder.

## Project Structure

```
wsn_project/
│
├── src/                       # Source code for WSN simulation
│   ├── __init__.py             # Python package marker
│   ├── network_setup.py        # Node initialization
│   ├── energy_model.py         # Energy consumption model
│   ├── leach_protocol.py       # LEACH protocol
│   ├── cro_protocol.py         # CRO protocol (placeholder)
│   ├── benchmarking.py         # Simulation and benchmarking
│   ├── visualization.py        # Network visualization
│   ├── edge_deployment.py      # Raspberry Pi Pico edge deployment (TFLite Micro)
│   └── main.py                 # Main script to run the simulation and model deployment
│
├── model/                     
│   └── model_quantized.tflite  # Quantized model file for Raspberry Pi Pico
│
├── data/                       
│   └── results.csv             # Simulation results for analysis
│
├── outputs/                    
│   ├── topology_plot.png       # Network topology visualization
│   └── benchmarking_plot.png   # Benchmarking results plot
│
├── requirements.txt            # Dependencies required for the project
└── README.md                   # Project description and instructions
```

6. **Project Dependencies (requirements.txt):**
```
networkx
numpy
matplotlib
scikit-learn
keras
```
