from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_aer import AerSimulator

def get_noisy_simulator() -> tuple[AerSimulator, NoiseModel]:
    """
    Instantiates an AerSimulator configured with the realistic noise model, basis gates,
    and coupling map properties of the 127-qubit IBM Brisbane hardware device.
    """

    #load the real hardware snapshot
    fake_backend = FakeBrisbane()

    #extract the full hardware noise profile
    noise_model = NoiseModel.from_backend(fake_backend)

    #build a high-performance local simulator matching the hardware characteristics 
    noisy_simulator = AerSimulator.from_backend(fake_backend)

    return noisy_simulator, noise_model

if __name__ == "__main__":
    sim, model = get_noisy_simulator()
    print("Successfully initialized Noise Modeling Layer")
    print(f"Target Backend Basis Gates: {sim.operation_names}")
    print(f"Noise model configured for: {len(model.noise_qubits)} qubits")