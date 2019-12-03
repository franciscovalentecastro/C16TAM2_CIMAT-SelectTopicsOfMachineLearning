# Based on Pytorch Torchvision Github
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/coco.py

from torchvision.datasets.vision import VisionDataset
from torchvision import transforms
from torchvision.utils import make_grid

from PIL import Image
import os
import os.path
import torch
import numpy as np
from pprint import pprint

from imshow import *


class CocoKeypoints(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_
        Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that
            takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that
            takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes
            input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, args, transform=None,
                 target_transform=None, transforms=None):
        super(CocoKeypoints, self).__init__(root, transforms,
                                            transform,
                                            target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        catIds = self.coco.getCatIds(catNms=['person'])
        self.ids = list(sorted(self.coco.getImgIds(catIds=catIds)))
        self.transform = transform

        # Keypoints parameter
        self.sigma = args.sigma
        self.target_type = 'gaussian'
        self.num_joints = 17
        self.image_size = np.array(args.image_shape)
        self.heatmap_size = np.array(args.image_shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object
                   returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        # Extract bounding box
        print(type(target[0]))
        if len(target[0]) > 0:
            bbox = torch.tensor(target[0]['bbox'])
            print(bbox)
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            print(x, y, w, h)

            # imshow(grid)
            plt.imshow(image)
            plt.axis('off')
            ax = plt.gca()
            coco.showAnns(target)

            image = image.crop((x, y, x + w, y + h))

        # Transform image
        img = self.transform(image)
        image_shape = tuple(img.shape[1:3])

        grid = make_grid(img, nrow=4, padding=2, pad_value=1)
        imshow(grid)

        # Exctract keypoints
        keypoints = torch.tensor(target[0]['keypoints'])
        keypoints = keypoints.reshape(17, 3)
        tmp_keypoints = keypoints.clone()

        # Crop keypoints
        keypoints[:, 0] = keypoints[:, 0] - x
        keypoints[:, 1] = keypoints[:, 1] - y

        # Rescale keypoints
        keypoints[:, 0] = ((tmp_keypoints[:, 1] * self.heatmap_size[0]) /
                           image_shape[0])
        keypoints[:, 1] = ((tmp_keypoints[:, 0] * self.heatmap_size[1]) /
                           image_shape[1])

        # If keypoints are missing
        if tuple(keypoints.shape) != (17, 3):
            keypoints = torch.zeros((17, 3), dtype=torch.float)

        # Generate heatmaps
        trgt, trgt_weight = self.generate_target(keypoints)
        target_torch = torch.tensor(trgt)

        print(target_torch.shape)
        targets_slice = target_torch.sum(dim=0, keepdim=True)
        grid = make_grid(targets_slice, nrow=4, padding=2, pad_value=1)
        imshow(grid)

        return img, target_torch

    def __len__(self):
        return len(self.ids)

    # Based on Github of Original Paper https://arxiv.org/pdf/1804.06208.pdf
    # https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/dataset/JointsDataset.py
    def generate_target(self, joints):  # , joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints[:, 2]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[0],
                               self.heatmap_size[1]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or \
                   ul[1] >= self.heatmap_size[1] or \
                   br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) /
                           (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_x[0]:img_x[1], img_y[0]:img_y[1]] = \
                        g[g_x[0]:g_x[1], g_y[0]:g_y[1]]

        return target, target_weight
