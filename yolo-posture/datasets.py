import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class VOC2007(Dataset):
    def __init__(self, image_shape):
        # Print message
        print('Started loading VOC2007.')

        # Set dataset transform
        transform = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor()
        ])

        # Load images
        image_path = './VOC2007/JPEGImages'

        self.images = torchvision.datasets.ImageFolder(
            root=image_path, transform=transform)

        # YOLO cell dimension
        S = 7

        # YOLO annotations path
        yolo_path = './VOC2007/yolo1_train_7/'

        if os.path.exists('y_real.npy'):
            # Load npy array
            y_real = np.load('y_real.npy')
        else:
            # Helping lists
            y_real = []

            # Travese all files in folder
            filenames = [yolo_path + instance_path
                         for instance_path in sorted(os.listdir(yolo_path))]
            y_real = np.array([np.loadtxt(f, comments='#').reshape((-1, S, S))
                               for f in filenames])

            # Save npy array
            np.save('y_real.npy', y_real)

        self.annotations = y_real

        # Print message
        print('VOC2007 was successfully loaded.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, _ = self.images[idx]
        target = torch.tensor(self.annotations[idx])

        # Only the following classes
        # bicycle :  1
        # bus     :  5
        # car     :  6
        # person  : 14

        # Tensor indexes
        bycicle_idx = 6
        bus_idx = 10
        car_idx = 11
        person_idx = 19

        # Indexes
        clss_idx = [0, 1, 2, 3, 4]
        clss_idx = clss_idx + [bycicle_idx, bus_idx, car_idx, person_idx]

        # Subset tensor
        target = target[clss_idx, :, :].transpose(1, 2)

        return image, target
