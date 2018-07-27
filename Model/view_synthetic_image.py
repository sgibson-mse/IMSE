import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def plot_image(image):
    plt.figure(1)
    plt.imshow(image, cmap='gray')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    return

filename1 = '/home/sgibson/PycharmProjects/IMSE/Model/synthetic_image2_32x32.hdf'
image1 = load_image(filename1)
image1 = np.array(image1)

plt.figure()
plt.imshow(image1)
plt.colorbar()
plt.show()
