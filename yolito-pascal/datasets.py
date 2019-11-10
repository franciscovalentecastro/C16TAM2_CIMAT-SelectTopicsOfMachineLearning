import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class VOC2007(Dataset):
    def __init__(self, image_shape, subset='train'):
        # Set dataset transform
        transform = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor()
        ])

        # Path to subset definiton
        if subset == 'train':
            subset_path = './VOC2007/ImageSets/Layout/train.txt'
        elif subset == 'val':
            subset_path = './VOC2007/ImageSets/Layout/val.txt'
        elif subset == 'test':
            subset_path = './VOC2007/ImageSets/Layout/test.txt'

        # Load images
        image_path = './VOC2007/JPEGImages'

        self.images = torchvision.datasets.ImageFolder(
            root=image_path, transform=transform)

        # YOLO cell dimension
        S = 7

        # YOLO annotations path
        yolo_path = './VOC2007/yolo1_train_7/'

        # Helping lists
        # classes = []
        y_real = []
        # file_names = []

        # Travese all files in folder
        for idx, instance_path in enumerate(os.listdir(yolo_path)):
            with open(yolo_path + instance_path, 'r') as f:
                y_file = np.loadtxt(f, comments='#')
                y_file = y_file.reshape((-1, S, S))

            y_real.append(torch.tensor(y_file))

            # file_names.append(f)
            # classes.append(y_file[5:].sum(axis=(1, 2)))

        self.annotations = y_real

        print(len(self.annotations))
        print(len(self.images))

        # Print message
        print('VOC2007 was successfully loaded !')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, _ = self.images[idx]
        target = self.annotations[idx]

        # Only the following classes
        # bycicle :  1
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
