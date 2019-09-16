import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
