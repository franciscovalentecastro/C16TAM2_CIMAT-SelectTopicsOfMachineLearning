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


def imshow_bboxes(images, targets, args):

    print(images.shape)
    print(targets.shape)

    # Necessary values
    img_w = images.shape[3]
    img_h = images.shape[2]
    cell_w = img_w / 7
    cell_h = img_h / 7

    print(img_w)
    print(img_h)
    print(cell_w)
    print(cell_h)

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
    for idx in range(args.batch_size):
        # Get slices
        # c = targets[idx][0]
        y = targets[idx][1]
        x = targets[idx][2]
        w = targets[idx][3]
        h = targets[idx][4]
        c1 = targets[idx][5]
        c2 = targets[idx][6]
        c3 = targets[idx][7]
        c4 = targets[idx][8]

        # Get indexes of found classes
        found = c1 + c2 + c3 + c4
        print(found.shape)
        print(found)

        found = found.nonzero()
        print(found.shape)
        print(found)

        if found.shape[0] == 0:
            continue

        # Get xy cell indexes
        px, py = found.split(1, 1)
        px = px.squeeze(1)
        py = py.squeeze(1)

        print(px)
        print(py)

        # Pick found classes
        y = y[px, py]
        x = x[px, py]
        w = w[px, py]
        h = h[px, py]

        print(x.shape)
        print(y.shape)
        print(w.shape)
        print(h.shape)

        print(x)
        print(y)
        print(w)
        print(h)

        for jdx in range(len(y)):
            w_bb = (img_h * h[jdx]).item()
            h_bb = (img_w * w[jdx]).item()
            y_bb = (cell_h * (py[jdx] + y[jdx])).item()
            x_bb = (img_w * idx + cell_w * (px[jdx] + x[jdx])).item()

            print(x_bb)
            print(y_bb)
            print(w_bb)
            print(h_bb)

            plt.gca().add_patch(Rectangle((x_bb - w_bb / 2,
                                           y_bb - h_bb / 2),
                                          w_bb, h_bb,
                                          linewidth=1,
                                          edgecolor='r',
                                          facecolor='none'))
            plt.text(x_bb - w_bb / 2, y_bb + h_bb / 2, 'lalala')

    # Save image
    # filename = "imgs/savefig_{}.png".format(image_counter)
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # Show image
    plt.show()
