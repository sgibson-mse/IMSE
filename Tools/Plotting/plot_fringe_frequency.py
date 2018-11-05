import numpy as np
import matplotlib.pyplot as plt

from Tools.Plotting.graph_format import plot_format

cb = plot_format()

def calc_fringe_frequency(displacer_cut_angle, focal_lengths):

    """
    Calculate the number of pixels per fringe for a given cut angle and focal length lens
    :param displacer_cut_angle: Angle of optical axis of the displacer (0-90 degrees)
    :return: Number of pixels per fringe on the sensor
    """

    pixel_size = 20*10**-6

    pixels_per_fringe = np.zeros((len(displacer_cut_angle), len(focal_lengths)))

    for j in range(len(focal_lengths)):
        for i in range(len(displacer_cut_angle)):
            waves_per_pixel = (2*np.pi*(3*10**-3)*0.117*np.sin(2*displacer_cut_angle[i]*(np.pi/180.)))/(1.61*(focal_lengths[j])*(660*10**-9))
            waves_per_m = waves_per_pixel*pixel_size/(2*np.pi)
            pixels_per_fringe[i,j] = 1/waves_per_m
            print(pixels_per_fringe[i,j])

    return pixels_per_fringe

displacer_cut_angle = np.arange(1,89,1)
focal_lengths = np.array([50 * 10 ** -3, 85 * 10 ** -3])
pixels_per_fringe = calc_fringe_frequency(displacer_cut_angle, focal_lengths)

plt.figure()
for i in range(len(focal_lengths)):
    plt.plot(pixels_per_fringe[:,i],  label='$f$='+str(focal_lengths[i]*1000)+'mm', color=cb[i])
plt.legend()
plt.xlim(30,60)
plt.ylim(8,20)
plt.xlabel('Displacer Cut Angle $\Theta$ (Deg.)')
plt.ylabel('$\#$ Pixels per fringe')
plt.show()
