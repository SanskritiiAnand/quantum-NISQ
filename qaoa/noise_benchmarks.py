from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

#load ideal metrics from max-cut.py
try:
    with open("qaoa_ideal_metrics.json", "r") as f:
        data = json.load(f)
    True_Bitstring = data["true_bitstring"]
    ideal_metrics = data["ideal_metrics"]
    print(f"Ideal baseline data loaded, target state: |{True_Bitstring}>\n")
except FileNotFoundError:
    print("Error: qaoa_ideal_metrics.json NOT FOUND, run max_cut.py first")
    exit()

#Rebuild same circuits for noise testing
num_nodes = 6 
np.random.seed(42) #must match max_cut.py exactly!
G= nx.complete_graph(num_nodes)
for (u,v) in G.edges():
    G[u][v]['weight'] = np.random.normal(0, 1.0)

pauli_list = [("ZZ", [u,v], 0.5 * G[u][v]['weight']) for (u,v) in G.edges()]
H_C = SparsePauliOp.from_sparse_list(pauli_list, num_qubits=num_nodes)

ansatz = QAOAAnsatz(cost_operator=H_C, reps=2)
test_params = [0.5, 0.25, 0.3, 0.15]
qaoa_circuit = ansatz.assign_parameters(test_params)
qaoa_circuit.measure_all()

#set up noisy backend & transpilation
print(f"Initializing FakeBrisbane noise profile & running transpiler...")
backend = AerSimulator.from_backend(FakeBrisbane())

pm= generate_preset_pass_manager(optimization_level=3, backend=backend) #level=3 helps route dense connections onto heavy-hex hardware
transpiled_ckt = pm.run(qaoa_circuit)

#noisy simulation
shot_budgets = [10, 100, 512, 1024, 4096]
noisy_results_data = {}

sampler= SamplerV2.from_backend(backend)

print("\n--- Running noisy shot budget convergence analysis ---")
for shots in shot_budgets:
    pub= (transpiled_ckt, None, shots)
    job= sampler.run([pub])
    result = job.result()[0]

    counts = result.data.meas.get_counts()

    true_match_counts = counts.get(True_Bitstring, 0)
    success_prob = true_match_counts / shots
    noisy_results_data[str(shots)]= success_prob

    print(f"Shot Budget: {shots:4d} | Found target bitstring {true_match_counts:3d} | Success Probability: {success_prob * 100:.2f}%")

#performance comparison plot
x_shots = [int(s) for s in ideal_metrics.keys()]
y_ideal = [ideal_metrics[str(s)] for s in x_shots]
y_noisy = [noisy_results_data[str(s)] for s in x_shots]

plt.figure(figsize=(9, 5.5))
plt.plot(x_shots, y_ideal, label= "Ideal (Statevector)", marker = 'o', color='#1f77b4', linewidth= 2.5)
plt.plot(x_shots, y_noisy, label= "Noisy (FakeBrisbane Simulator)", marker= 'x', color= '#d62728', linestyle='--', linewidth=2.5)

plt.xscale('log')
plt.xlabel("Shot Budget (Log scale)", fontsize= 11)
plt.ylabel("Success Probability of Max-Cut State", fontsize= 11)
plt.title(f"QAOA Shot Budget Convedrgence: Ideal vs Noisy ({num_nodes}- Qubit SK Model)", fontsize= 13, fontweight='bold')
plt.xticks(x_shots, labels=[str(s) for s in x_shots])
plt.grid(True, which='both', ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("qaoa_shot_convergence_comparison.png", dpi=300, bbox_inches='tight')

print(f"\nOpening plot window...")
plt.show()