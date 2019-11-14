# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models

from pprint import pprint


class loss_yolo():
    def __init__(self, args):
        self.counter = 0

        # Network to train
        self.device = args.device

        # Number of bboxes per cell
        self.bboxes = args.bboxes

        # Input image shape
        self.img_w, self.img_h = args.image_shape
        self.cell_w, self.cell_h = (self.img_w / 7.0, self.img_h / 7.0)

        # Test iou
        # a = torch.tensor([[0.4, 0.4, 0.05, 0.1]]).unsqueeze(2).unsqueeze(3)
        # b = torch.tensor([[0.6, 0.6, 0.05, 0.1]]).unsqueeze(2).unsqueeze(3)
        # pprint(a.shape)
        # pprint(a.numpy())
        # pprint(b.shape)
        # pprint(b.numpy())

        # iou = self.IOU_bbox(a, b)
        # pprint(iou.shape)
        # pprint(iou.numpy)

    def IOU_bbox(self, bbox_1, bbox_2):
        # Unpack first bbox elements
        x1 = bbox_1[:, 0] * self.cell_w
        y1 = bbox_1[:, 1] * self.cell_h
        w1 = bbox_1[:, 2] * self.img_w
        h1 = bbox_1[:, 3] * self.img_h

        # Unpack second bbox elements
        x2 = bbox_2[:, 0] * self.cell_w
        y2 = bbox_2[:, 1] * self.cell_h
        w2 = bbox_2[:, 2] * self.img_w
        h2 = bbox_2[:, 3] * self.img_h

        # Calculate first bbox corners
        b1_x1 = x1 - (w1 / 2.0)
        b1_x2 = x1 + (w1 / 2.0)
        b1_y1 = y1 - (h1 / 2.0)
        b1_y2 = y1 + (h1 / 2.0)

        # Calculate second bbox corners
        b2_x1 = x2 - (w2 / 2.0)
        b2_x2 = x2 + (w2 / 2.0)
        b2_y1 = y2 - (h2 / 2.0)
        b2_y2 = y2 + (h2 / 2.0)

        # Find intersection box
        x1_int = torch.max(b1_x1, b2_x1)
        x2_int = torch.min(b1_x2, b2_x2)
        y1_int = torch.max(b1_y1, b2_y1)
        y2_int = torch.min(b1_y2, b2_y2)

        # Calculate intersection
        inter = (x2_int - x1_int) * (y2_int - y1_int)
        positive_inter = (inter > 0.0).float()
        inter = inter * positive_inter

        # Calculate iou
        union = (w1 * h1) + (w2 * h2) - inter
        iou = inter / union

        # pprint('inter {}'.format(inter.shape))
        # pprint('inter {}'.format(inter.item()))
        # pprint('union {}'.format(union.shape))
        # pprint('union {}'.format(union.item()))
        # pprint('iou {}'.format(iou.shape))
        # pprint('iou {}'.format(iou.item()))

        # pprint(positive_inter.shape)
        # pprint(positive_inter.item())
        # pprint(iou.shape)
        # pprint(iou.item())

        # import matplotlib.pyplot as plt
        # plt.xlim(0, self.cell_w)
        # plt.ylim(0, self.cell_h)
        # plt.plot([x1], [y1], 'v', color='red')
        # plt.plot([x2], [y2], 'v', color='blue')
        # plt.plot([b1_x1, b1_x2, b1_x2, b1_x1, b1_x1],
        #          [b1_y1, b1_y1, b1_y2, b1_y2, b1_y1], '.-', color='red')
        # plt.plot([b2_x1, b2_x2, b2_x2, b2_x1, b2_x1],
        #          [b2_y1, b2_y1, b2_y2, b2_y2, b2_y1], '.-', color='blue')
        # plt.plot([x1_int, x2_int, x2_int, x1_int, x1_int],
        #          [y1_int, y1_int, y2_int, y2_int, y1_int], '.-', color='green')
        # plt.show()

        return iou

    def IOU(self, outputs, targets):
        # Calculate IOU
        if self.bboxes == 1:
            # Index of bbox elements
            bbox_idx = [1, 2, 3, 4]

            # Get only bbox elements
            bbox = outputs[:, bbox_idx]
            bbox_trg = targets[:, bbox_idx]

            # Calculate iou of target and output
            iou = self.IOU_bbox(bbox, bbox_trg)

            return iou

        elif self.bboxes == 2:
            # Index of bbox elements
            bbox1_idx = [1, 2, 3, 4]
            bbox2_idx = [6, 7, 8, 9]

            # Get only bbox elements
            bbox_1 = outputs[:, bbox1_idx]
            bbox_2 = outputs[:, bbox2_idx]
            bbox_trg = targets[:, bbox1_idx]

            iou_1 = self.IOU_bbox(bbox_1, bbox_trg)
            iou_2 = self.IOU_bbox(bbox_2, bbox_trg)

            return iou_1, iou_2

    def loss(self, outputs, targets):
        # Needed parameters
        batch_size = outputs.shape[0]
        lmbd_coord = 5
        lmbd_noobj = .5

        # Caclulate iou and select best bbox
        iou = self.IOU(outputs, targets)

        # Pick best bbox
        if self.bboxes == 2:
            iou_1, iou_2 = iou

            # print(iou_1.shape)
            # print(iou_2.shape)

            # print((iou_1 >= iou_2).shape)
            # print((iou_2 > iou_1).shape)

            comp_1 = torch.nonzero((iou_1 >= iou_2))
            comp_2 = torch.nonzero((iou_2 > iou_1))

            # print(comp_1.shape)
            # print(comp_2.shape)

            b1, x1, y1 = (comp_1[:, 0], comp_1[:, 1], comp_1[:, 2])
            b2, x2, y2 = (comp_2[:, 0], comp_2[:, 1], comp_2[:, 2])

            t_outputs = torch.zeros([batch_size, 9, 7, 7])

            # print(b1.shape)
            # print(x1.shape)
            # print(y1.shape)

            # print(b2.shape)
            # print(x2.shape)
            # print(y2.shape)

            # print(t_outputs[b1, :, x1, y1].shape)
            # print(t_outputs[b2, :, x2, y2].shape)

            # print(outputs[b1, slice1_idx, x1, y1].shape)
            # print(outputs[b2, 5:, x2, y2].shape)

            t_outputs[b1, :5, x1, y1] = outputs[b1, :5, x1, y1]
            t_outputs[b1, 5:, x1, y1] = outputs[b1, 10:, x1, y1]
            t_outputs[b2, :, x2, y2] = outputs[b2, 5:, x2, y2]

            outputs = t_outputs
            print(outputs.shape)

            # Save correct iou
            iou = torch.max(iou_1, iou_2)
            print(iou.shape)

        # Get output elements
        c_out = outputs[:, 0]
        y_out = outputs[:, 1]
        x_out = outputs[:, 2]
        w_out = torch.sqrt(outputs[:, 3])
        h_out = torch.sqrt(outputs[:, 4])
        c1_out = outputs[:, 5]
        c2_out = outputs[:, 6]
        c3_out = outputs[:, 7]
        c4_out = outputs[:, 8]

        # Ger target elements
        y_trgt = targets[:, 1]
        x_trgt = targets[:, 2]
        w_trgt = torch.sqrt(targets[:, 3])
        h_trgt = torch.sqrt(targets[:, 4])
        c1_trgt = targets[:, 5]
        c2_trgt = targets[:, 6]
        c3_trgt = targets[:, 7]
        c4_trgt = targets[:, 8]
        c_trgt = (c1_trgt + c2_trgt + c3_trgt + c4_trgt) * iou

        # Calculate loss
        obj = c1_trgt + c2_trgt + c3_trgt + c4_trgt
        noobj = torch.ones(batch_size, 7, 7).to(self.device) - obj

        self.counter += 1
        if self.counter % 10 == 0:
            print('w_trgt', w_trgt.shape)
            print(w_trgt[0])
            print('w_out', w_out.shape)
            print(w_out[0])
            print('h_trgt', h_trgt.shape)
            print(h_trgt[0])
            print('h_out', h_out.shape)
            print(h_out[0])
            print('iou', iou.shape)
            print(iou[0])
            print('c_trgt', c_trgt.shape)
            print(c_trgt[0])
            print('c_out', c_out.shape)
            print(c_out[0])
            print('obj', obj.shape)
            print(obj[0])

            print('prod', (obj * (c_out - c_trgt)).shape)
            print((obj * (c_out - c_trgt))[0])


        # Loss function calculation
        loss = lmbd_coord * ((obj * (x_out - x_trgt)) ** 2).sum()
        loss += lmbd_coord * ((obj * (y_out - y_trgt)) ** 2).sum()
        loss += lmbd_coord * ((obj * (w_out - w_trgt)) ** 2).sum()
        loss += lmbd_coord * ((obj * (h_out - h_trgt)) ** 2).sum()
        loss += ((obj * (c_out - c_trgt)) ** 2).sum()
        loss += ((obj * (c1_out - c1_trgt)) ** 2).sum()
        loss += ((obj * (c2_out - c2_trgt)) ** 2).sum()
        loss += ((obj * (c3_out - c3_trgt)) ** 2).sum()
        loss += ((obj * (c4_out - c4_trgt)) ** 2).sum()
        loss += lmbd_noobj * (noobj * (c_out - c_trgt) ** 2).sum()

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
