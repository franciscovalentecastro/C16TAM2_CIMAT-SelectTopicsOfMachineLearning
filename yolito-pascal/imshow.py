import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# functions to show an images
def imshow(img):
    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    # Show image
    plt.show()


def plot_bboxes(images, bboxes, args, color='r'):
    # Necessary values
    img_w = images.shape[3]
    img_h = images.shape[2]
    cell_w = img_w / 7
    cell_h = img_h / 7

    for idx in range(args.batch_size):

        # Get slices
        c, y, x, w, h, c1, c2, c3, c4 = bboxes[idx].split(1, 0)
        t_var = [c, y, x, w, h, c1, c2, c3, c4]
        c, y, x, w, h, c1, c2, c3, c4 = [x.squeeze(0) for x in t_var]

        # Get indexes of found classes
        found = c1 + c2 + c3 + c4
        found = found.nonzero()
        if found.shape[0] == 0:
            continue

        # Get xy cell indexes
        px, py = found.split(1, 1)
        px = px.squeeze(1)
        py = py.squeeze(1)

        # Pick found classes
        c = c[px, py]
        y = y[px, py]
        x = x[px, py]
        w = w[px, py]
        h = h[px, py]
        c1 = c1[px, py]
        c2 = c2[px, py]
        c3 = c3[px, py]
        c4 = c4[px, py]

        for jdx in range(len(y)):
            if c[jdx] > .8:
                # Get class named
                class_nms = ['bicycle', 'bus', 'car', 'person']
                class_pred = [c1[jdx].item(), c2[jdx].item(),
                              c3[jdx].item(), c4[jdx].item()]
                maxpos = class_pred.index(max(class_pred))
                pred_name = class_nms[maxpos]

                w_bb = (img_h * h[jdx]).item()
                h_bb = (img_w * w[jdx]).item()
                y_bb = (cell_h * (py[jdx] + y[jdx])).item()
                x_bb = (img_w * idx + cell_w * (px[jdx] + x[jdx])).item()

                plt.gca().add_patch(Rectangle((x_bb - w_bb / 2,
                                               y_bb - h_bb / 2),
                                              w_bb, h_bb,
                                              linewidth=1,
                                              edgecolor=color,
                                              facecolor='none'))
                plt.text(x_bb - w_bb / 2, y_bb + h_bb / 2, pred_name,
                         color='white', fontsize=10)


def imshow_bboxes(images, targets, args, predictions=None):

    # Necessary values
    img_w = images.shape[3]
    img_h = images.shape[2]
    cell_w = img_w / 7
    cell_h = img_h / 7

    # Plot cell grid
    dx, dy = int(cell_w), int(cell_h)

    # Modify the image to include the grid
    images[:, :, :, ::dy] = 0
    images[:, :, ::dx, :] = 0

    # Create images grid
    grid = torchvision.utils.make_grid(images,
                                       nrow=args.batch_size,
                                       padding=0)

    # Plot images
    npimg = grid.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    # Plot bounding boxes
    plot_bboxes(images, targets, args)

    # Check if predictions were provided
    if predictions is not None:
        plot_bboxes(images, predictions, args, color='g')

    # Get current figure
    fig = plt.gcf()
    fig.tight_layout()
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    buf = fig.canvas.buffer_rgba()
    l, b, w, h = fig.bbox.bounds

    # The array needs to be copied, because the underlying buffer
    # may be reallocated when the window is resized.
    img = np.frombuffer(buf, np.uint8).copy()
    img = torch.tensor(img.reshape(int(h), int(w), 4)).permute(2, 0, 1)

    # Drop borders of figure
    img = img[:, 350:650, :]

    if args.plot:
        # Show image
        plt.show()

    return img
