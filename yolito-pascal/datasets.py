import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Tsukuba(Dataset):
    def __init__(self, transform=transforms.ToTensor(), train=True):
        if train:
            path = './tsukuba/train/'
        else:
            path = './tsukuba/test/'

        self.right = torchvision.datasets.ImageFolder(
            root=path + 'right', transform=transform)

        self.left = torchvision.datasets.ImageFolder(
            root=path + 'left', transform=transform)

        self.disp = torchvision.datasets.ImageFolder(
            root=path + 'disp',
            transform=transforms.Compose([transforms.Grayscale(),
                                          transform]))

    def __len__(self):
        return len(self.right)

    def __getitem__(self, idx):
        left, _ = self.left[idx]
        right, _ = self.right[idx]
        disp, _ = self.disp[idx]

        return left, right, disp
