import matplotlib.pyplot as plt
import numpy as np


image_counter = 0


# functions to show an image
def imshow(img):
    global image_counter
    image_counter += 1

    npimg = img.detach().numpy()
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    filename = "imgs/savefig_{}.png".format(image_counter)
    plt.savefig(filename)
    plt.show()
