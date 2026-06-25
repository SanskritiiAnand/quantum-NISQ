# NISQ-Era Hybrid Quantum Algorithms Baseline Suite

A production-grade implementation of hybrid quantum-classical algorithms evaluated under realistic hardware noise layers using modern Qiskit frameworks. 
This repository benchmarks algorithm convergence behavior and optimization metrics against classical exact solutions.

## Project Structure

```text
quantum-NISQ/
├── vqe/
│   ├── molecules.py          # Active-space Hamiltonian factory & exact baseline solver
│   ├── noise_engine.py       # Hardware noise cloning layer (IBM Brisbane snapshot)
│   └── run_vqe.py            # VQE orchestrator featuring multi-strategy SciPy loops
├── qaoa/                     # (In Progress) Combinatorial Max-Cut optimization layer
├── .gitignore
└── requirements.txt
```
## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/SanskritiiAnand/quantum-NISQ.git
cd quantum-NISQ

-- If git not pre-installed: Click 'Code' at top of the page, download ZIP folder, extract the files and open them in terminal/IDE --
```

### 2. Configure virtual environment
```bash
Windowws:
python -m venv venv
venv\Scripts\activate
Mac/Linux:
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run master engine
```bash
python main.py
```
