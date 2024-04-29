import pennylane as qml
from pennylane.math import fidelity
from torch import nn
import torch


dev = qml.device("lightning.qubit", wires=4)

zero_label, one_label = torch.zeros(size=(2, 2)), torch.zeros(size=(2, 2))
zero_label[1, 1], one_label[0, 0] = 1., 1.


@qml.qnode(dev, interface='torch')
def qcircuit(inputs, quantum_params):
    """
    inputs : (4*3+1,)
    quantum_params: (3,)
    """
    data, y = inputs[:-1].reshape(4, 3), inputs[-1:]
    y = one_label if y else zero_label

    for i, row in enumerate(data):
        qml.RX(row[0], wires=i)
        qml.RY(row[1], wires=i)
        qml.RX(row[2], wires=i)

    qml.ISWAP(wires=[1, 0])
    qml.ISWAP(wires=[2, 1])
    qml.ISWAP(wires=[3, 2])

    for i, row in enumerate(data):
        qml.RY(row[0], wires=i)
        qml.RX(row[1], wires=i)
        qml.RY(row[2], wires=i)

    qml.ISWAP(wires=[3, 2])
    qml.ISWAP(wires=[2, 1])
    qml.ISWAP(wires=[1, 0])

    qml.RX(quantum_params[0], wires=0)
    qml.RY(quantum_params[1], wires=0)
    qml.RX(quantum_params[2], wires=0)

    return qml.expval(qml.Hermitian(y, wires=0))


class QCNN(nn.Module):
    def __init__(self):
        super(QCNN, self).__init__()

        self.weight_shapes = {
            "quantum_params": (3,)
        }
        
        self.qlayer = qml.qnn.TorchLayer(qcircuit, self.weight_shapes)
        
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=3, 
            stride=2
        )

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, 0, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            if isinstance(m, qml.qnn.TorchLayer):
                pi_over_2 = torch.acos(torch.zeros(1)).item()
                nn.init.uniform_(m.quantum_params, 0, pi_over_2)


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
