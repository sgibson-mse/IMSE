from scipy.io import readsav
import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from Analysis.KSTAR_data.demodulate_kstar_image import demodulate_image

file_directory = '/mnt/cifs/sam/'
structure_file = 'str_9034.sav'

def read_sav(file_directory, file):
    data = readsav(str(file_directory + file))
    return data

d = read_sav(file_directory, structure_file)

tags = ['sh', 'camera', 'camangle', 'flencam', 'rotate', 'spe', 'tif', 'tifrec', 'tifpre', 'tdms', 'tdmspre',
        'tdmspost', 'path', 'tree', 'mdsplusnode', 'mdsplusseg',
        'flc0mark', 'flct0', 'flc0per', 'flc0invert', 'flc0endt', 'flc0endstate', 'flc1mark', 'flc1t0',
        'flc1per', 'cellno', 't0', 't0proper', 'nskip', 'dt', 'tstampavail'
                                                              'roil', 'roir', 'roib', 'roit', 'binx', 'biny',
        'xbin', 'polariser', 'calno', 'calfile', 'calfilebg', 'lambdanm', 'mapping', 'comment', 'mapstr',
        'pixsizemm', 'ivec',
        'nfr', 'pinfoflc']

flc_info = d['str'].pinfoflc

flc0_t = flc_info[0][0][0][0][0]
flc0_v = flc_info[0][0][0][0][1]
flc_switch = 4 #number of images taken in the same FLC state
n_frames = np.arange(100,102,1)

phases = np.zeros((2160,2560,2))

filenames = []
for i in range(len(n_frames)):
    filenames.append('im_9034_' + str(n_frames[i]) + '.sav')
    image_flc = read_sav(file_directory, filenames[i])
    image = image_flc['im']

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.clim(10,7500)
    plt.show()

    phase, contrast = demodulate_image(image)
    phases[:,:,i] = phase

# doppler_phase = (phases[:,:,0] + phases[:,:,1])/2.
# polarisation_angle = (np.average(phases[:,:,0:4], axis=2) - doppler_phase)/2.
#
# doppler_phase_2 = (phases[:,:,4] + phases[:,:,5])/2.
# polarisation_angle_2 = (np.average(phases[:,:,4:9], axis=2) - doppler_phase)/2.
#
# for i in range(24):
#     doppler_phase = phases[:,:,i] + phases[:,:,i+1] / 2
#     pol = (np.average(phases[:,:,i:i+4], axis =2 ) - doppler_phase)/2





# doppler_phase = np.zeros((2160,2560,6))
#
# for i in range(len(phases[0,0,:])-5):
#     doppler_phase[:,:,int(i/4)] = (phases[:,:,i] + phases[:,:,i+1])/2.
#     polarisation_angle = (np.average(phases[:,:,i:i+4], axis=2) - doppler_phase[:,:,int(i/4)])/2.
#     i += 4
#
# plt.figure()
# for i in range(len(doppler_phase[-2])):
#     plt.plot(doppler_phase[1080,:,i], label='set {}'.format(i))
# plt.legend()
# plt.show()
#
# plt.figure()
# for i in range(len(polarisation_angle[-2])):
#     plt.plot(polarisation_angle[1080,:,i], label='set {}'.format(i))
# plt.legend()
# plt.show()



# plt.figure()
# plt.plot(flc0_t, flc0_v)
# plt.show()