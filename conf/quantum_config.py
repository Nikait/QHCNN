import torch
import pennylane as qml


n_qubits: int = 12

dev = qml.device("lightning.qubit", wires=n_qubits)

zero_label, one_label = torch.zeros(size=(2, 2)), torch.zeros(size=(2, 2))
zero_label[1, 1], one_label[0, 0] = 1., 1.


add_noise: bool = False
# probabilities of applying depolarising channel
# after data uploading
P1: float = 0.005
# after ISWAP gate
P2: float = 0.005
