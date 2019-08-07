
import matplotlib.pyplot as plt
import numpy as np
import animatplot as amp

import xarray as xr

def load_image(filename):
    return xr.open_dataset(filename)

def change_datatype(data):
    return np.array(data['data']).astype(np.uint16)

def fft_2d(image):
    return np.fft.fft2(image, axes=(0,1))

def box_filter(image, x_size, y_size, centre):

    x = len(image[0,:])
    y = len(image[:,0])

    mask = np.zeros((np.shape(image)))

    mask[int(centre[1]-y_size/2): int(centre[1] + y_size/2), int(centre[0]- (x_size/2)): int(centre[0] + x_size/2)] = 1.

    return mask

def demodulate(timeslices):

    gamma = np.zeros((1424,1200, len(timeslices)))

    for i, timeslices in enumerate(timeslices):

        print(i)
        filename = '/home/sam/Desktop/Projects/AUG/IMSE/data/3536/35368/RAW/IMAGE_{}.nc'.format(timeslices)

        data = load_image(filename)

        image = change_datatype(data)

        plt.figure()
        plt.imshow(image)
        cs = plt.colorbar()
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        cs.ax.set_ylabel('Intensity (counts/ms)', rotation=90)
        plt.clim(0,5000)
        plt.show()

        fft = fft_2d(image)
        image_fft = np.fft.fftshift(fft)

        levels = np.arange(7,11,1)

        phi_dfilter = box_filter(image, x_size=100, y_size=100, centre=[694,683])

        phid_plus_phi_s = box_filter(image, x_size = 100, y_size=100, centre=[660,573])

        phid_minus_phi_s = box_filter(image, x_size =100, y_size=100, centre=[728,784])

        #Apply Filters
        image_phid = phi_dfilter * image_fft

        image_pos = phid_plus_phi_s * image_fft

        image_neg = phid_minus_phi_s * image_fft

        image_pos_ifft = np.fft.ifft2(image_pos)
        image_neg_ifft = np.fft.ifft2(image_neg)
        image_phid_ifft = np.fft.ifft2(image_phid)

        gamma[:,:,i] = np.arctan(4*(abs(image_pos_ifft)*abs(image_neg_ifft))/abs(image_phid_ifft)**2)/2

        #gamma_datasets.append(gamma)

    #gamma = xr.concat(gamma_datasets, dim='polarisation_angle')

    return gamma

def plot_fft(image_fft):

    plt.figure()
    plt.imshow(np.log10(abs(image_fft)))
    plt.colorbar()
    plt.clim(6,9)
    plt.show()

def plot_gamma(gamma):
    plt.figure()
    cs = plt.imshow(gamma*(180./np.pi), cmap='viridis')
    cs.cmap.set_under('white')
    cs.cmap.set_over('white')
    plt.colorbar()
    plt.clim(20,40)
    plt.show()

background = '/home/sam/Desktop/Projects/AUG/IMSE/data/3536/35368/RAW/IMAGE_0.nc'

timeslices = np.arange(800,870,1)
gamma = demodulate(timeslices)

block = amp.blocks.Imshow(gamma*(180./np.pi), t_axis=2, cmap='viridis', vmin=5, vmax=40)
plt.xlim(800,1200)
plt.ylim(150,500)
plt.title('Polarisation angle variation due to magnetic field Bt = -0.9T')
cs = plt.colorbar(block.im)
cs.cmap.set_over('white')
cs.cmap.set_under('white')
plt.gca().set_aspect('equal')
cs.ax.set_ylabel('Polarisation Angle $\gamma$ (Degrees)', rotation=90)
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
anim = amp.Animation([block], amp.Timeline(timeslices))

anim.controls()
# anim.save('faraday_rotation' + '.gif', writer='imagemagick')
plt.show()

#For usual shots, the box size and sensor size is:
#1760, 1960
#100,100, [1134,828]
#100,100, [1187, 969]
#100,100, [1081,694]