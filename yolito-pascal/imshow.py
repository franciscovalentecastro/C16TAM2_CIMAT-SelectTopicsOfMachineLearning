import numpy as np
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# functions to show an images
def imshow(img):
    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

    # Save image
    # filename = "imgs/savefig_{}.png".format(image_counter)
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

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

        for jdx in range(len(y)):
            if c[jdx] > .95:
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
                plt.text(x_bb - w_bb / 2, y_bb + h_bb / 2, 'lalala')


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

    # Save image
    # filename = "imgs/savefig_{}.png".format(image_counter)
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Show image
    plt.show()
