import pennylane as qml
from pennylane import numpy as np
from torch import nn
import torch

from conf.quantum_config import *
from conf.structured_config import Config


def noise_layer(prob: float) -> None:
    for j in range(n_qubits):
        if add_noise:
            # Depolarising channel
            if np.random.choice([1, 0], p=[prob/3, 1-prob/3]):
                qml.PauliX(wires=j)
            if np.random.choice([1, 0], p=[prob/3, 1-prob/3]):
                qml.PauliY(wires=j)
            if np.random.choice([1, 0], p=[prob/3, 1-prob/3]):
                qml.PauliZ(wires=j)


def iswap_layer(acseding: bool) -> None:
    for i in range(n_qubits-1):
        if not acseding:
            qml.ISWAP(wires=[i+1, i])
        else:
            qml.ISWAP(wires=[n_qubits-i-1, n_qubits-i-2])


@qml.qnode(dev, interface='torch')
def qcircuit(
    inputs: torch.Tensor, quantum_params: torch.Tensor
    ) -> torch.Tensor:
    """
    inputs : (6*6+1,)
    quantum_params: (3,)
    """

    data, y = inputs[:-1].reshape(6, 6), inputs[-1:]
    y = one_label if y else zero_label

    for i, row in enumerate(data):
        qml.RX(row[:3][0], wires=2*i)
        qml.RY(row[:3][1], wires=2*i)
        qml.RX(row[:3][2], wires=2*i)

        qml.RX(row[3:][0], wires=2*i+1)
        qml.RY(row[3:][1], wires=2*i+1)
        qml.RX(row[3:][2], wires=2*i+1)

    noise_layer(P1)
    iswap_layer(True)
    noise_layer(P2)

    qml.Barrier(wires=[0, 11])

    for i, row in enumerate(data):
        qml.RY(row[:3][0], wires=2*i)
        qml.RX(row[:3][1], wires=2*i)
        qml.RY(row[:3][2], wires=2*i)

        qml.RY(row[3:][0], wires=2*i+1)
        qml.RX(row[3:][1], wires=2*i+1)
        qml.RY(row[3:][2], wires=2*i+1)

    noise_layer(P1)
    iswap_layer(False)
    noise_layer(P1)

    qml.RX(quantum_params[0], wires=0)
    qml.RY(quantum_params[1], wires=0)
    qml.RX(quantum_params[2], wires=0)

    return qml.expval(qml.Hermitian(y, wires=0))


class QCNN(nn.Module):
    def __init__(self, cfg: Config):
        super(QCNN, self).__init__()

        self.__pi = 2 * torch.acos(torch.zeros(1)).item()

        self.weight_shapes = {
            "quantum_params": (3,)
        }
        
        self.qlayer = qml.qnn.TorchLayer(qcircuit, self.weight_shapes)
        
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=4, 
            stride=2
        )

        self.initialize_weights(cfg)


    def initialize_weights(self, cfg: Config) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, 0, cfg.train.conv_weight)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, cfg.train.bias)
            if isinstance(m, qml.qnn.TorchLayer):
                nn.init.uniform_(m.quantum_params, 0, cfg.train.quantum_weight)


    def forward(self, images: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.conv(images)
        loss = torch.FloatTensor([0])

        for i, image in enumerate(out):
            image = torch.flatten(image)
            inputs = torch.cat((image, y[i].reshape(1)), dim=0)
            out = self.qlayer(inputs)
            res = (1 - out) ** 2
            loss += res

        return loss / images.shape[0]


    def predict(self, images: torch.Tensor) -> torch.Tensor:
        predictions = torch.Tensor(size=(images.shape[0],))
        with torch.no_grad():
            out = self.conv(images)
            for i, image in enumerate(out):
                image = torch.flatten(image)
                input1 = torch.cat((image, torch.Tensor([0])), dim=0)
                input2 = torch.cat((image, torch.Tensor([1])), dim=0)
                f1, f2 = self.qlayer(input1), self.qlayer(input2)
                _, ind = torch.max(torch.stack((f1, f2)), 0)
                predictions[i] = ind

        return predictions
