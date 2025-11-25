# Quantum Network Protocol Design

A machine learning project for quantum network protocol design using PPO (Proximal Policy Optimization) agents for qubit movement optimization.

## Project Structure

```
├── data/
│   ├── machine_learning_agents/    # Trained ML agents
│   ├── machine_learning_logs/      # TensorBoard training logs (ignored by git)
│   ├── processed/                  # Processed data files
│   └── raw/                        # Raw data files
├── helper_functions/
│   ├── __init__.py
│   └── qubit_mover_2.py           # Core qubit mover implementation
├── notebooks/
│   └── qubit_mover2.ipynb         # Jupyter notebook for experimentation
├── tests/                          # Unit tests
└── requirements.txt               # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/noah-1019/QNT_protocol_design.git
cd QNT_protocol_design
```

2. Create and activate a virtual environment:
Make sure you are using Python 3.9 or higher.
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Add usage instructions here]

## Dependencies

- Python 3.9+
- PyTorch
- Stable Baselines3
- Gymnasium
- SymPy
- NumPy
- Matplotlib
- Pandas

## Training

The project uses PPO agents for training qubit movement strategies. Training logs are saved to `data/machine_learning_logs/` and can be visualized with TensorBoard.

## Contributing

[Add contribution guidelines if applicable]