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
