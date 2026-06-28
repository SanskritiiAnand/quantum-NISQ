from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json

#sherrington-kirpkpatrick model
num_nodes = 6
np.random.seed(42)  #for reproducible graph weights

G = nx.complete_graph(num_nodes)

for (u,v) in G.edges():
    G[u][v]['weight'] = np.random.normal(0, 1.0)

print(f"---- Generated SK Model with {num_nodes} nodes and {len(G.edges())} weighted edges ----")

#construct cost hamiltonian
pauli_list = []
for (u,v) in G.edges():
    weight = G[u][v]['weight']
    #edge interaction: Z on node u and Z on node v; H_C = 0.5 * sum(w_jj*Z_i*Z_j)
    pauli_list.append(("ZZ", [u,v], 0.5 * weight))

H_C = SparsePauliOp.from_sparse_list(pauli_list, num_qubits= num_nodes)
print(f"Successfully constructed Cost Hamiltonian H_C:\n{H_C}\n")

#classical baseline
H_matrix_diag = H_C.to_matrix().diagonal()
best_state_idx = np.argmin(np.real(H_matrix_diag))

True_Bitstring = format(best_state_idx, f'0{num_nodes}b') #convert index to padded bitstring representaion
print(f"Classical True MAx-Cut Minimum State: |{True_Bitstring}|")
print(f"Classical Optimal Minimum Energy Value: {np.real(H_matrix_diag[best_state_idx]): .4f}\n")

#ansatz and parameters
ansatz = QAOAAnsatz(cost_operator=H_C, reps=2) #total 4 parameter values
test_params = [0.5, 0.25, 0.3, 0.15]
qaoa_circuit = ansatz.assign_parameters(test_params)
qaoa_circuit.measure_all()

#executing looping ovre shot budgets
shot_budgets = [10, 100, 512, 1024, 4096]
ideal_results_data = {}

sampler = StatevectorSampler()

print("--- Running Ideal Shot Budget Convergence Analysis ---")
for shots in shot_budgets:
    pub = (qaoa_circuit, None, shots)
    job = sampler.run([pub])
    result = job.result()[0]

    counts = result.data.meas.get_counts()

    true_match_counts = counts.get(True_Bitstring, 0)
    success_prob = true_match_counts / shots
    ideal_results_data[str(shots)] = success_prob

    print(f"Shot Budget: {shots:4d} | Found Target Bitstring {true_match_counts:3d} | Success Probability: {success_prob*100:.2f}%")

with open("qaoa_ideal_metrics.json", "w") as f:
    json.dump({"true_bitstring": True_Bitstring, "ideal_metrics": ideal_results_data}, f)
print("\nGenerating visualization")
plt.figure(figsize=(7,6))
pos = nx.circular_layout(G)
edges = G.edges()
weights= [G[u][v]['weight'] for (u,v) in edges]

edge_colors= []
edge_widths= []
for w in weights: 
    if w>0:
        edge_colors.append('#2ca02c') 
    else:
        edge_colors.append( '#d62728')
    
    edge_widths.append(abs(w)*3)

nx.draw_networkx_nodes(G, pos, node_color='#9467bd', node_size=700)
nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors, alpha= 0.7)
nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')

plt.title(f"Sherrington-Kirkpatrick Model Topology ($K_{num_nodes}$)\n(Green = Positive weight, Red = Negative weight)", fontsize=12, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig("max_cut_graph.png", dpi=300)
plt.show()