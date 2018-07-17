import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hanning_window(image):
    h = np.hanning(len(image))
    b = np.ones((np.shape(image)))
    return b*h

def load_image(filename):
    image_file = pd.HDFStore(filename)
    print(image_file.keys())
    image = image_file['/a']
    return image

def apply_hanning(image):
    window = hanning_window(image)
    return window*image

images3 = load_image('ash_image_S34mm.hdf')
images3 = apply_hanning(images3)

image_nos3 = load_image('ash_image_noS3_savart4mm.hdf')
image_nos3 = apply_hanning(image_nos3)

plt.figure()
plt.subplot(121)
plt.imshow(np.log10(abs(np.fft.fftshift(np.fft.fft2(images3, axes=(1,0))))))
plt.gca().invert_xaxis()
plt.clim(11.5,12.5)
plt.colorbar()

plt.subplot(122)
plt.imshow(np.log10(abs(np.fft.fftshift(np.fft.fft2(image_nos3, axes=(1,0))))))
plt.gca().invert_xaxis()
plt.clim(11.5,12.5)
plt.colorbar()
plt.show()

plt.figure(2)
plt.subplot(121)
plt.imshow(images3, cmap='gray')
plt.gca().invert_xaxis()
plt.colorbar()

plt.subplot(122)
plt.imshow(image_nos3, cmap='gray')
plt.gca().invert_xaxis()
plt.colorbar()
plt.show()