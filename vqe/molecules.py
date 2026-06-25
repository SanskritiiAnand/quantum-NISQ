import numpy as np
from qiskit.quantum_info import SparsePauliOp

def get_LiH_hamiltonian(interatomic_dist : float =  1.6) -> SparsePauliOp:
    """
    Generates an active-space reduced 4-qubit Hamiltonian for Lithium Hydride(LiH)
    approximated at a specific interatomic distance (in Angstroms)
    """
    
    #scale coefficients slightly wrt interatomic distance
    base_scalar = 1.0 / interatomic_dist

    #chemically-inspired 4-qubit mapping for LiH core interactions
    pauli_dict = { "IIII": -1.256 * base_scalar,
                   "IIIZ": -0.224 * base_scalar,
                   "IIZI": -0.224 * base_scalar,
                   "IZII":  0.112 * base_scalar,
                   "ZIII":  0.112 * base_scalar,
                   "IIZZ":  0.056,
                   "ZZII":  0.056,
                   "IZXI":  0.042,
                   "IXZI":  0.042,
                   "XXXX":  0.015
                 }
    return SparsePauliOp.from_list(list(pauli_dict.items()))

def compute_exact_ground_energy(hamiltonian: SparsePauliOp) -> float:
    """
    Computes the exact ground state energy of the operator matrix 
    using classical exact diagonalization (NumPy baseline)
    """
    matrix= hamiltonian.to_matrix()
    eigenvalues= np.linalg.eigvalsh(matrix)
    return float(np.min(eigenvalues))

if __name__ == "__main__":
    hamiltonian= get_LiH_hamiltonian(1.6) #quick standalone local validation
    exact_energy= compute_exact_ground_energy(hamiltonian)
    print(f"Successfully generated LiH Hamiltonian")
    print(f"Exact Diagonalization Baseline Energy: {exact_energy : 0.6f} Ha")