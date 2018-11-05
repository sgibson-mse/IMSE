import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Tools.Plotting.graph_format import plot_format, plot_save
from mpl_toolkits.axes_grid1 import make_axes_locatable

cb = plot_format()

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def plot_zoom_image(image, filename):

    fig = plt.subplots(nrows=1, ncols=1)
    ax = plt.gca()

    im = plt.imshow(image[400:700,400:700])
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Intensity (ph/s)', labelpad=4)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.gca().invert_xaxis()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plot_save(filename)

def plot_image(image, filename):

    fig = plt.subplots(nrows=1, ncols=1)
    ax = plt.gca()

    im = plt.imshow(image)
    cbar = plt.colorbar(im, orientation='vertical')
    cbar.set_label('Intensity (ph/s)', labelpad=4)
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.gca().invert_xaxis()
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show()

    #plot_save(filename)


filename1 = '/home/sgibson/PycharmProjects/IMSE/Model/ideal_contrast.hdf'
filename2 = '/home/sgibson/PycharmProjects/IMSE/Model/ideal_phase.hdf'

image1 = load_image(filename1)
image1 = np.array(image1)

image2 = load_image(filename2)
image2 = np.array(image2)

plt.figure()
plt.plot(image1[16,:])
plt.show()


