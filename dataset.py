import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    @staticmethod
    def unison_shuffle(
            a: torch.Tensor, 
            b: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:

        assert len(a) == len(b), "lengths are not matched!"
        permutation = torch.randperm(len(a))

        return a[permutation], b[permutation]

    @staticmethod
    def transform_data(
            data: torchvision.datasets,
            width: int,
            height: int,
            min_len: int
        ) -> tuple[torch.Tensor, ...]:

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((width, height))
            ]
        )

        x_data = torch.Tensor(size=(2 * min_len, width, height))
        y_data = torch.Tensor(size=(2 * min_len,))

        zeros_labels = torch.Tensor([0 for i in range(min_len)])
        ones_labels = torch.Tensor([1 for i in range(min_len)])

        zeros = list(
            map(
                lambda x: transform(x[0]),
                filter(lambda x: x[1] == 0, data)
            )
        )
        ones = list(
            map(
                lambda x: transform(x[0]),
                filter(lambda x: x[1] == 1, data)
            )
        )
        
        torch.cat([*zeros[:min_len], *ones[:min_len]], out=x_data)
        torch.cat([zeros_labels, ones_labels], out=y_data)

        return x_data, y_data
    

    def __init__(
            self, 
            load_dir: str, 
            width: int, 
            height: int, 
            min_len: int, 
            train: bool
        ):

        self.__length = 2 * min_len

        loaded_data = torchvision.datasets.MNIST(
            load_dir, 
            download=True,
            train=train
        )

        self.x_data, self.y_data = self.transform_data(
            loaded_data, width, height, min_len
        )

        self.x_data, self.y_data = self.unison_shuffle(
            self.x_data, self.y_data
        )

        self.x_data = self.x_data.reshape(shape=(self.__length, 1, width, height))


    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """
        returns:
            tensor: (1, width, height), tensor: (1,)
        """
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        return self.__length
