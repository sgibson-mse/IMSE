import numpy as np

#Read a .dat binary file which contains images from the IMSE diagnostic system.

def load_ashbinary(filename, FLC):

    """
    :param filename: Binary filename. Binary file contains a header with 3 16 bit unassigned integers denoting number of pixels in x and y and the number of frames taken.
    :param FLC: If the FLC is on, FLC must be true, as there will be double the number of frames (one for each polarisation state)
    :return: Array containing two images for each rotation.
    """

    with open(filename,'rb') as f:

        header = np.fromfile(f, dtype=np.int16, count=2)

        nx = header[0]
        ny = header[1]

        header_32bit = np.fromfile(f, dtype=np.int32, count=1)

        n_frames = header_32bit[0]

        header_floating = np.fromfile(f, dtype=np.float32, count=1)

        step = header_floating[0]

        header_32 = np.fromfile(f, dtype=np.int32, count=1)

        theta0 = header_32[0]

        header_32bit_final = np.fromfile(f, dtype=np.int32, count=1)

        status = header_32bit_final[0]

        if FLC:
            #If the FLC is present, there will be 2 frames corresponding to two separate polarisation states.
            n_frames = n_frames*2
        else:
            pass

        data = np.fromfile(f, dtype=np.uint16)

        #Order must be fortran-like to preserve the memory layout

        images = data.reshape(nx, ny, n_frames, order='F')

    return images, step, theta0

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#calculate the offset between the polariser angle vs what we put in via rotary stage
#
# filename = '/home/sam/Desktop/Projects/IMSE-MSE/Analysis/sam_11.dat'
# images, step, theta0 = load_ashbinary(filename, FLC=False)
#
# n_frames = 37
# theta = np.linspace(theta0,360.,n_frames)*(np.pi/180.)
#
# def fit(theta,a):
#     return 0.25 + 0.25*np.cos(2*(-a+(3*np.pi/4.)-theta))
#
# bg, bgstep, bgtheta0 = load_ashbinary('/home/sam/Desktop/Projects/IMSE-MSE/Analysis/sam_12.dat', FLC=False)
#
# image_slice =  images[1080,1180,:].astype(np.int32) - bg[1080,1180,:].astype(np.int32)
#
# guess = [0.82]
# popt, pcov = curve_fit(fit, theta, image_slice, p0=guess)
#
# y_fit = image_slice*fit(theta,popt)
#
# print(popt*(180./np.pi))
#
# plt.figure()
# plt.plot(theta*(180/np.pi), images[1080,1280,:]/np.max(images[1028,1280,:]))
# plt.plot(theta*(180./np.pi), y_fit/np.max(y_fit), '--', label='fit')
# plt.legend()
# plt.ylabel('I')
# plt.xlabel('Polariser angle (degrees)')
# plt.show()
