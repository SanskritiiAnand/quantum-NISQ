import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.circuit.library import TwoLocal
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2
from molecules import get_LiH_hamiltonian, compute_exact_ground_energy
from noise_engine import get_noisy_simulator

def run_vqe_optimization():
    print("---- Initializing VQE Workflow ----")

    hamiltonian = get_LiH_hamiltonian(interatomic_dist= 1.6)
    exact_energy = compute_exact_ground_energy(hamiltonian)
    num_qubits = hamiltonian.num_qubits

    #using hardware efficient Ry-Rz parameterized ansatz with  linear entanglement
    ansatz = TwoLocal(num_qubits=num_qubits, rotation_blocks=['ry','rz' ],
                      entanglement_blocks= 'cz', entanglement='linear', reps=1,
                      insert_barriers= True)
    num_params = ansatz.num_parameters

    backend, noise_model = get_noisy_simulator()

    #transpile the abstract ansatz & observable to match the backend layout
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled_ansatz = pass_manager.run(ansatz)
    transpiled_hamiltonian = hamiltonian.apply_layout(transpiled_ansatz.layout)

    #initialize modern V2 Estimator with noise parameters
    estimator = EstimatorV2.from_backend(backend)
    
    #containers to store convergence paths for plotting
    cobyla_history = []
    powell_history = []

    #objective function that the classical SciPy optimizer calls iteratively
    def cost_function(params, history_list):
        pub= (transpiled_ansatz, transpiled_hamiltonian, params) #bind current parameters to the transpiled ansatz
        job= estimator.run([pub]) #EstimattorV2 takes a tuple of (circuit, observables, parameter_values)
        result= job.result()[0]

        #extract the expectation value (<H>)
        energy = float(result.data.evs)
        history_list.append(energy)
        return energy
    
    #run COBYLA
    print("\n Running COBYLA optimizer...")
    initial_point = np.zeros(num_params) #start from zero angles
    minimize(cost_function, initial_point, args=(cobyla_history,), method='COBYLA', options={'maxiter': 40})

    #run Powell
    print("\n Running Powell optimizer...")
    minimize(cost_function, initial_point, args=(powell_history,), method='Powell', options={'maxiter': 25})

    print("\n ----Optimization Complete----")
    print(f"Exact Target Energy: {exact_energy: .6f}Ha")
    print(f"Final COBYLA Energy: {cobyla_history[-1]: .6f}Ha")
    print(f"Final Powell Energy: {powell_history[-1]: .6f}Ha")

    plt.figure(figsize=(10,6))
    plt.plot(cobyla_history, label='COBYLA', color='tab:blue', lw=2)
    plt.plot(spsa_history, label='POWELL', color='tab:orange', lw=2)
    plt.axhline(y= exact_energy, color='tab:red', linestyle='--', label='Exact Energy Baseline')
    plt.title("VQE Convergence Comparison on Noisy LiH System", fontsize=14)
    plt.xlabel("Optimizer Iterations", fontsize=12)
    plt.ylabel("Energy(Hartrees)", fontsize=12)
    plt.grid(True, linestyle= ':', alpha=0.6)
    plt.legend(fontsize=11)

    plt.savefig('vqe_convergence_plot.png', dpi=300, bbox_inches='tight')
    print("Convergence plot generated and saved")
    plt.show()

if __name__ == "__main__":
    run_vqe_optimization()
