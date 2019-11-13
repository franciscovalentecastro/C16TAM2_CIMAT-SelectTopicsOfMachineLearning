# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models

import matplotlib.pyplot as plt


class loss_yolo():
    def __init__(self, args):
        # Network to train
        self.device = args.device

        # Number of bboxes per cell
        self.bboxes = args.bboxes

        # Input image shape
        self.img_w, self.img_h = args.image_shape
        self.cell_w, self.cell_h = (self.img_w / 7.0, self.img_h / 7.0)

    def IOU_ind_bbox(self, bbox_1, bbox_2):
        x1 = bbox_1[0].item() * self.cell_w
        y1 = bbox_1[1].item() * self.cell_h
        w1 = bbox_1[2].item() * self.img_w
        h1 = bbox_1[3].item() * self.img_h

        x2 = bbox_2[0].item() * self.cell_w
        y2 = bbox_2[1].item() * self.cell_h
        w2 = bbox_2[2].item() * self.img_w
        h2 = bbox_2[3].item() * self.img_h

        # print(x1, y1, w1, h1)
        # print(x2, y2, w2, h2)

        b1_x1 = x1 - (w1 / 2.0)
        b1_x2 = x1 + (w1 / 2.0)
        b1_y1 = y1 - (h1 / 2.0)
        b1_y2 = y1 + (h1 / 2.0)

        b2_x1 = x2 - (w2 / 2.0)
        b2_x2 = x2 + (w2 / 2.0)
        b2_y1 = y2 - (h2 / 2.0)
        b2_y2 = y2 + (h2 / 2.0)

        # print(b1_x1, b1_x2, b1_y1, b1_y2)
        # print(b2_x1, b2_x2, b2_y1, b2_y2)

        if b1_x2 < b2_x1 or b2_x2 < b1_x1 or \
           b1_y2 < b2_y1 or b2_y2 < b1_y1:
            return 0

        x1_int = max(b1_x1, b2_x1)
        x2_int = min(b1_x2, b2_x2)
        y1_int = max(b1_y1, b2_y1)
        y2_int = min(b1_y2, b2_y2)

        # print(x1_int, x2_int, y1_int, y2_int)

        inter = (x2_int - x1_int) * (y2_int - y1_int)
        union = (w1 * h1) + (w2 * h2) - inter

        # print(inter / union)
        # if inter / union != 0:
        #     plt.plot([b1_x1, b1_x1, b1_x2, b1_x2, b1_x1],
        #              [b1_y1, b1_y2, b1_y2, b1_y1, b1_y1], '.-', color='red')
        #     plt.plot([b2_x1, b2_x1, b2_x2, b2_x2, b2_x1],
        #              [b2_y1, b2_y2, b2_y2, b2_y1, b2_y1], '.-', color='blue')
        #     # plt.ylim(0, self.img_h)
        #     # plt.xlim(0, self.img_w)
        #     plt.show()

        return inter / union

    def IOU(self, outputs, targets):
        batch_size = outputs.shape[0]

        # Calculate IOU
        if self.bboxes == 1:
            # print('bboxes ', self.bboxes)

            bbox1_idx = [1, 2, 3, 4]
            iou = torch.zeros([batch_size, 7, 7])

            for idx in range(batch_size):
                for jdx in range(7):
                    for kdx in range(7):
                        bbox = outputs[idx, bbox1_idx, jdx, kdx].reshape(4)
                        bbox_trg = targets[idx, bbox1_idx, jdx, kdx].reshape(4)

                        iou[idx, jdx, kdx] = self.IOU_ind_bbox(bbox, bbox_trg)
                        # print('iou ', idx, jdx, kdx, iou[idx, jdx, kdx].item())

            # print(iou.shape)
            # print(iou)

            return iou.reshape(batch_size * 49), outputs

        elif self.bboxes == 2:
            bbox1_idx = [1, 2, 3, 4]
            bbox2_idx = [6, 7, 8, 9]

            iou_1 = torch.zeros([batch_size, 7, 7])
            iou_2 = torch.zeros([batch_size, 7, 7])

            # print(iou_1.shape)
            # print(iou_2.shape)

            for idx in range(batch_size):
                for jdx in range(7):
                    for kdx in range(7):
                        bbox_1 = outputs[idx, bbox1_idx, jdx, kdx].reshape(4)
                        bbox_2 = outputs[idx, bbox2_idx, jdx, kdx].reshape(4)
                        bbox_trg = targets[idx, bbox1_idx, jdx, kdx].reshape(4)

                        iou_1[idx, jdx, kdx] = self.IOU_ind_bbox(bbox_1,
                                                                 bbox_trg)
                        iou_2[idx, jdx, kdx] = self.IOU_ind_bbox(bbox_2,
                                                                 bbox_trg)

            slice1_idx = [0, 1, 2, 3, 4, 10, 11, 12, 13]
            slice2_idx = [5, 6, 7, 8, 9, 10, 11, 12, 13]

            t_outputs = torch.zeros([batch_size, 9, 7, 7])
            for idx in range(batch_size):
                for jdx in range(7):
                    for kdx in range(7):
                        if iou_1[idx, jdx, kdx].item() > iou_2[idx, jdx, kdx].item():
                            t_outputs[idx, :, jdx, kdx] = outputs[idx,
                                                                  slice1_idx,
                                                                  jdx, kdx]
                        else:
                            t_outputs[idx, :, jdx, kdx] = outputs[idx,
                                                                  slice2_idx,
                                                                  jdx, kdx]
            outputs = t_outputs

            return torch.max(iou_1, iou_2).reshape(batch_size * 49), t_outputs

    def loss(self, outputs, targets):
        # Needed parameters
        batch_size = outputs.shape[0]
        lmbd_coord = 5
        lmbd_noobj = .5

        # Caclulate iou and select best bbox
        iou, outputs = self.IOU(outputs, targets)

        # print(outputs.shape)
        # input()

        # Get output elements
        c_out = outputs[:, 0].reshape(batch_size * 49) * iou
        y_out = outputs[:, 1].reshape(batch_size * 49)
        x_out = outputs[:, 2].reshape(batch_size * 49)
        w_out = torch.sqrt(outputs[:, 3]).reshape(batch_size * 49)
        h_out = torch.sqrt(outputs[:, 4]).reshape(batch_size * 49)
        c1_out = outputs[:, 5].reshape(batch_size * 49)
        c2_out = outputs[:, 6].reshape(batch_size * 49)
        c3_out = outputs[:, 7].reshape(batch_size * 49)
        c4_out = outputs[:, 8].reshape(batch_size * 49)

        # Ger target elemtns
        c_trgt = targets[:, 0].reshape(batch_size * 49)
        y_trgt = targets[:, 1].reshape(batch_size * 49)
        x_trgt = targets[:, 2].reshape(batch_size * 49)
        w_trgt = torch.sqrt(targets[:, 3]).reshape(batch_size * 49)
        h_trgt = torch.sqrt(targets[:, 4]).reshape(batch_size * 49)
        c1_trgt = targets[:, 5].reshape(batch_size * 49)
        c2_trgt = targets[:, 6].reshape(batch_size * 49)
        c3_trgt = targets[:, 7].reshape(batch_size * 49)
        c4_trgt = targets[:, 8].reshape(batch_size * 49)

        # Calculate loss
        obj = c1_trgt + c2_trgt + c3_trgt + c4_trgt
        noobj = torch.ones(batch_size * 49).to(self.device) - obj

        loss = (lmbd_coord * obj * (x_out - x_trgt) ** 2).sum()
        loss += (lmbd_coord * obj * (y_out - y_trgt) ** 2).sum()
        loss += (lmbd_coord * obj * (w_out - w_trgt) ** 2).sum()
        loss += (lmbd_coord * obj * (h_out - h_trgt) ** 2).sum()
        loss += (obj * (c_out - c_trgt) ** 2).sum()
        loss += (obj * (c1_out - c1_trgt) ** 2).sum()
        loss += (obj * (c2_out - c2_trgt) ** 2).sum()
        loss += (obj * (c3_out - c3_trgt) ** 2).sum()
        loss += (obj * (c4_out - c4_trgt) ** 2).sum()
        loss += (lmbd_noobj * noobj * (c_out - c_trgt) ** 2).sum()

        return loss / batch_size


class YOLO(nn.Module):

    def __init__(self, args):
        super(YOLO, self).__init__()
        # Number of bboxes per cell
        self.bboxes = args.bboxes

        # Classifier input dimension
        self.lin_inpt = 512 * (args.image_shape[0] // 32) * \
            (args.image_shape[1] // 32)

        self.vgg = models.vgg16(pretrained=True)

        self.linear = nn.Sequential(nn.Linear(self.lin_inpt,
                                              self.lin_inpt // 2),
                                    nn.ReLU(),
                                    nn.Linear(self.lin_inpt // 2,
                                              (5 * self.bboxes + 4) * 7 * 7),
                                    nn.Sigmoid())

    def forward(self, x):
        x1 = self.vgg.features(x)
        x1 = x1.view(-1, self.lin_inpt)
        x2 = self.linear(x1)
        y = x2.view([-1, (5 * self.bboxes + 4), 7, 7])

        return y
