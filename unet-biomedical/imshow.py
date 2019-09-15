import matplotlib.pyplot as plt
import numpy as np

figure_number = 0


# functions to show an image
def imshow(img):
    global figure_number
    figure_number += 1

    img = img / 2 + 0.5     # from range (0,1) to (-1,1)
    npimg = img.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),
               interpolation='nearest')
    plt.savefig("figure_%d.png" % figure_number,
                bbox_inches='tight')
    plt.show()
