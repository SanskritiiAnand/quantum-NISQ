from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

try:
    with open("qaoa_ideal_metrics.json", "r") as f:
        data = json.load(f)
    True_Bitstring = data["true_bitstring"]
    ideal_metrics = data["ideal_metrics"]
    print(f"Ideal baseline records loaded. target stae: |{True_Bitstring}>\n")
except FileNotFoundError:
    print("Error: qaoa_ideal_metrics.json not found! run max_cut.py first")
    exit()

num_nodes = 6
np.random.seed(42)  
G= nx.complete_graph(num_nodes)
for (u,v) in G.edges():
    G[u][v]['weight'] = np.random.normal(0, 1.0)

pauli_list = [("ZZ", [u,v], 0.5 * G[u][v]['weight']) for (u,v) in G.edges()]
H_C = SparsePauliOp.from_sparse_list(pauli_list, num_qubits = num_nodes)

ansatz = QAOAAnsatz(cost_operator=H_C, reps=2)
test_params = [0.5, 0.25, 0.3, 0.15] #initial unoptimised angles

#Noisy environment setup
print("Initializing FakeBrisbane noisee model and transpiling ansatz...")
backend = AerSimulator.from_backend(FakeBrisbane())

pm= generate_preset_pass_manager(optimization_level=3, backend=backend)
isa_ansatz = pm.run(ansatz)
isa_observables = H_C.apply_layout(isa_ansatz.layout)

#Native SPSA
estimator = EstimatorV2.from_backend(backend)
loss_history = []

def cost_function(params):
    pub= (isa_ansatz, isa_observables, params, 0.015625)
    job= estimator.run([pub])
    energy= job.result()[0].data.evs
    loss_history.append(float(energy))
    return float(energy)

def spsa_optimize(cost_func, x0, maxiter=40, c=0.1, a=0.2, alpha=0.602, gamma=0.101):
    theta= np.array(x0, dtype=float)
    print("---Starting hardware-native SPSA Optimization Loop---")

    for k in range(1, maxiter + 1):
        ak= a/ (k+1)**alpha
        ck= c/ (k+1)**gamma

        delta= np.random.choice([-1,1], size=len(theta)) #simultaneous random perturbation vector
        theta_plus= theta + ck * delta
        theta_minus= theta - ck * delta
        y_plus= cost_func(theta_plus)
        y_minus= cost_func(theta_minus)

        gradient= (y_plus - y_minus)/(2 * ck * delta)

        theta -= ak*gradient
        if k % 5 == 0 or k == 1:
            print(f"Iteration {k:2d}/{maxiter} | Estinmated Energy: {loss_history[-1]:.4f}")

    return theta
    
optimal_params = spsa_optimize(cost_function, test_params, maxiter=40)

print(f"Optimal Noisy Variational Angles Found: {optimal_params}\n")

#SampleV2 execution with optimized angles
opt_ckt= ansatz.assign_parameters(optimal_params)
opt_ckt.measure_all()
transpiled_opt_ckt= pm.run(opt_ckt)
shot_budgets= [10, 100, 512, 1024, 4096]
optimized_noisy_metrics= {}

sampler= SamplerV2.from_backend(backend)

print("--- Running Noisy Shot Budget Convergence Analysis(SPSA optimized) ---")
for shots in shot_budgets:
    pub= (transpiled_opt_ckt, None, shots)
    job= sampler.run([pub])
    result= job.result()[0]

    counts= result.data.meas.get_counts()
    true_match_counts= counts.get(True_Bitstring, 0)
    success_prob= true_match_counts / shots
    optimized_noisy_metrics[str(shots)] = success_prob

    print(f"Shot Budget: {shots:4d} | Target Hits: {true_match_counts:4d} | Success Probability: {success_prob * 100:.2f}%")

#fetch unoptimised noisy data from noise_benchmarks.py
try: 
    from noise_benchmarks import noisy_results_data as unopt_noisy_metrics
except ImportError:
    print("\nWarning: Could not import 'noisy_results_data' from 'noise_benchmarks.py' ")
    print("Using flat dummy baseline for plot fallback")
    unopt_noisy_metrics = {str(s): 0.05 for s in shot_budgets}
    
#generate triple-line performance plot
x_shots= [int(s) for s in ideal_metrics.keys()]
y_ideal= [ideal_metrics[str(s)] for s in x_shots]
y_unopt_noisy= [unopt_noisy_metrics[str(s)] for s in x_shots]
y_opt_noisy= [optimized_noisy_metrics[str(s)] for s in x_shots]

plt.figure(figsize=(10,6))
plt.plot(x_shots, y_ideal, label="Ideal Backend (Unoptimized Baseline)", marker='o', color='#1f77b4', linewidth=2)
plt.plot(x_shots, y_unopt_noisy, label="Noisy Backend (Unoptimized Raw)", marker='x', color='#d62728', linestyle='--', linewidth=2)
plt.plot(x_shots, y_opt_noisy, label="Noisy Backend (Hardware-Native SPSA Optimized)", marker='s', color='#2ca02c', linewidth=2.5)

plt.xscale('log')
plt.xlabel("Shot Budget (Log Scale)", fontsize=11)
plt.ylabel("Success Probability of Target Max-Cut State", fontsize=11)
plt.title(f"QAOA Noise Remediation Profile via Native SPSA ({num_nodes}-Qubit SK Model)", fontsize=12, fontweight='bold')
plt.xticks(x_shots, labels=[str(s) for s in x_shots])
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10, loc="upper left")

plt.savefig("qaoa_spsa_noise_remediation.png", dpi=300, bbox_inches='tight')
print("\nSuccess!")
plt.show()