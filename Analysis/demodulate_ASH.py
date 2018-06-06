#External Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from Analysis.peak_find import indexes
from scipy.optimize import curve_fit
from scipy.stats.mstats import gmean
from scipy import signal

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#Internal imports
from Analysis.read_binary_ASH import load_ashbinary

def get_images(filename):
    images, step, theta0 = load_ashbinary(filename=filename, FLC=False)
    return images, step, theta0

def single_image(images, frame):
    image = images[:,:,frame]
    return image.T

def subtract_background(image, background):
    image_background = image.astype(np.int32) - (background[:,:,0].T).astype(np.int32)
    image_background[np.where(image_background<0)] = 0
    return image_background

def fft_2D(image):
    return np.fft.fft2(image, axes=(0,1))

def box_filter(image, x_size, y_size, centre):
    x = len(image[0,:])
    y = len(image[:,0])

    mask = np.zeros((np.shape(image)))

    mask[centre[1]:centre[1] + y_size, centre[0]:centre[0] + x_size] = 1.

    return mask

def apply_mask(image, mask):
    return image*mask

def unwrap(phase_array):

    y_pix, x_pix = np.shape(phase_array)
    phase_contour = -np.unwrap(phase_array[int(np.round(y_pix / 2)), :])

    # sequentially unwrap image columns:
    phase_uw_col = np.zeros_like(phase_array)

    for i in range(0, x_pix):
        phase_uw_col[:, i] = np.unwrap(phase_array[:, i])

    phase_contour = phase_contour + phase_uw_col[int(np.round(y_pix / 2)), :]
    phase_0 = np.tile(phase_contour, [y_pix, 1])
    phase_uw = phase_uw_col - phase_0

    # wrap image centre into [-pi, +pi] (assumed projection of optical axis onto detector)
    y_centre_idx = np.round((np.size(phase_uw, 0) - 1) / 2).astype(np.int)
    x_centre_idx = np.round((np.size(phase_uw, 1) - 1) / 2).astype(np.int)
    phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]

    if phase_uw_centre > 0:
        while abs(phase_uw_centre) > np.pi:
            phase_uw -= 2*np.pi
            phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]
    else:
        while abs(phase_uw_centre) > np.pi:
            phase_uw += 2*np.pi
            phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]

    return phase_uw

def apply_filters(image, image_fft):
    #filter the displacer phase
    phi_dfilter = box_filter(image, x_size=41, y_size=25, centre=[1255,1010])

    #filter phi_d + phi_s
    phid_plus_phi_s = box_filter(image, x_size = 41, y_size=25, centre=[1305,1010])

    #filter phi_d - phi_s
    phid_minus_phi_s = box_filter(image, x_size = 41, y_size=25, centre=[1210,1010])

    #apply filters
    image_phid = phi_dfilter * image_fft

    image_pos = phid_plus_phi_s * image_fft

    image_neg = phid_minus_phi_s * image_fft

    # plt.figure()
    # plt.imshow(abs(np.log10(image_fft)))
    # plt.colorbar()
    # # plt.clim(8,10)
    # plt.imshow(abs(phi_dfilter), alpha=0.3)
    # #plt.imshow(abs(phid_minus_phi_s), alpha=0.3)
    # #plt.imshow(abs(phid_plus_phi_s), alpha=0.3)
    # plt.show()

    return image_phid, image_pos, image_neg

def find_average_angle(gamma):
    center = [int(np.shape(gamma)[0]/2), int(np.shape(gamma)[1]/2)]
    theta_ave = np.average(gamma[center[0],:])
    return theta_ave

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im,g, mode='valid')
    return(improc)

def demodulate(background, frame):
    #get one frame
    image = single_image(images, frame=frame)

    #subtract background
    image_bg = subtract_background(image, background)

    # plt.figure()
    # plt.imshow(image_bg)
    # plt.colorbar()
    # #plt.clim(0,4000)
    # # plt.xlim(1000,2000)
    # # plt.ylim(1000,2000)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    #take fft
    fft_image = fft_2D(image_bg)

    #shift so DC in center
    image_fft = np.fft.fftshift(fft_image)

    #apply filters for each carrier frequency (phi_d, phi_d + phi_s, phi_d - phi_s

    image_phid, image_pos, image_neg = apply_filters(image, image_fft)

    #IFFT
    image_pos_ifft = np.fft.ifft2(image_pos)
    image_neg_ifft = np.fft.ifft2(image_neg)
    image_phid_ifft = np.fft.ifft2(image_phid)

    # plt.figure()
    # plt.title('A(+ +- )')
    # plt.plot(abs(image_neg_ifft[1000,:])+abs(image_pos_ifft[1000,:]))
    # # plt.colorbar()
    # plt.show()
    #
    # plt.figure()
    # plt.title('A(+ + )')
    # plt.plot(np.arctan((abs(image_pos_ifft[1000,:])+abs(image_neg_ifft[1000,:]))/abs(image_phid_ifft[1000,:])))
    # # plt.colorbar()
    #
    # plt.show()
    #
    # plt.figure()
    # plt.title('A(+ 0)')
    # plt.plot(abs(image_phid_ifft[1000,:]))
    # # plt.colorbar()
    #
    # plt.show()

    image_total_ifft = abs(image_pos_ifft) + abs(image_neg_ifft)

    #Extract phase from image
    gamma = np.arctan(abs(image_phid_ifft)/abs(image_total_ifft))/2.

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.imshow(gamma*(180./np.pi))
    # cs = plt.colorbar()
    # #plt.clim(30,45)
    # cs.set_label('Polarisation angle (degrees)', rotation=90)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(gamma[200,:]*(180/np.pi))
    # plt.ylabel('Polarisation Angle (Degrees)')
    # plt.xlabel('Pixel number')
    # plt.show()


    #Find average angle across the center of the image
    theta_ave = find_average_angle(gamma)

    # print(theta_ave*180/np.pi)
    # plt.figure()
    # plt.imshow(gamma*(180./np.pi))
    # plt.colorbar()
    # plt.show()

    return theta_ave

filename ='/home/sam/Desktop/Projects/IMSE-MSE/Analysis/sam_13.dat'
background ='/home/sam/Desktop/Projects/IMSE-MSE/Analysis/sam_12.dat'

images, step, theta0 = get_images(filename)
background, background_step, background_theta0 = get_images(background)
n_frames = 37
polarisation = np.zeros(n_frames)
for i in range(n_frames):
    polarisation[i] = demodulate(background, frame=i)

offset = 49.3479565 #degrees
offset_rad = offset*(np.pi/180.)
input_theta = np.linspace(theta0,360.,n_frames)*(np.pi/180.)
ideal = abs(np.arctan(np.sin((2*input_theta+offset_rad))/np.cos((2*input_theta+offset_rad))))/2.

plt.figure()
plt.plot(input_theta*(180./np.pi)+offset, polarisation*(180./np.pi), '--', color='red', label='Measured $\Theta$')
plt.plot(input_theta*(180./np.pi)+offset, ideal*(180./np.pi), color='black', label='Ideal $\Theta$')
plt.xlabel('Input polarisation (degrees)')
plt.ylabel('Output polarisation (degrees)')
plt.legend()
plt.show()

plt.figure()
plt.plot(input_theta*180./np.pi, (polarisation-ideal)*180/np.pi)
plt.xlabel('Input polarisation (degrees)')
plt.ylabel('$\delta \gamma$ Ideal - Measured (degrees)')
plt.show()