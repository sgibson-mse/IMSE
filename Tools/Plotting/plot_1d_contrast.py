import matplotlib.pyplot as plt
import numpy as np

from Tools.Plotting.graph_format import plot_format
from Tools.calculate_1d_contrast import get_stokes_components, find_optimal_crystal

plot_format()

# Find nearest function, useful for plotting

def find_nearest(array, value):
    if value < array.min() or value > array.max():
        raise IndexError(
            "Requested value is outside the range of the data. The range of the data is from {}m to {}m".format(
                array.min(), array.max()))

    index = np.searchsorted(array, value, side="left")

    if (value - array[index]) ** 2 < (value - array[index + 1]) ** 2:
        return index
    else:
        return index + 1

def plot_contrast_contourplot(contrasts, delay_thickness, major_radius):

    print(contrasts.shape)

    pp,ll = np.meshgrid(delay_thickness,major_radius)
    levels=np.arange(0,0.5,0.1)
    plt.figure()
    im = plt.pcolormesh(ll,pp*1000,contrasts,shading='gouraud')
    cbar = plt.colorbar(im)
    cbar.set_label('Contrast', rotation=90)
    CS = plt.contour(ll,pp*1000,contrasts, colors='black', levels=levels)
    plt.ylim(5,25)
    plt.clabel(CS, inline=1)
    plt.xlabel('Major Radius (m)')
    plt.ylabel('Delay plate thickness (mm)')
    plt.show()

    #Seperate contourplot
    # plt.figure()
    # plt.title('Displacer plate 3mm, $\Theta$=45$^{\circ}$')
    # CS =plt.contour(ll,pp*1000,contrasts, levels=levels)
    # plt.clabel(CS, inline=1, fontsize=10)
    # cbar2 = plt.colorbar()
    # plt.xlabel('Major Radius [m]')
    # plt.ylabel('Delay plate thickness [mm]')
    # cbar2.set_label('Contrast', rotation=90)
    # plt.show()

    idx1 = find_nearest(delay_thickness, 0.014)
    idx2 = find_nearest(delay_thickness, 0.015)
    idx3 = find_nearest(delay_thickness, 0.016)

    plt.figure()
    plt.plot(major_radius, contrasts[:,idx1], label='L = 14mm')
    plt.plot(major_radius, contrasts[:,idx2], label='L = 15mm')
    plt.plot(major_radius, contrasts[:,idx3], label='L = 16mm')
    plt.legend()
    plt.xlabel('Major radius (m)')
    plt.show()

def plot_contrast_cut_angle(contrasts, displacer_cut_angle, major_radius):

    pp,ll = np.meshgrid(displacer_cut_angle,major_radius)
    levels=np.arange(0,1,0.05)
    plt.figure()
    im = plt.pcolormesh(ll,pp*180./np.pi,contrasts,shading='gouraud')
    cbar = plt.colorbar(im)
    cbar.set_label('Contrast', rotation=90)
    CS = plt.contour(ll,pp*180./np.pi,contrasts, colors='black', levels=levels)
    plt.clabel(CS, inline=1)
    plt.xlabel('Major Radius (m)')
    plt.ylabel('Displacer Cut Angle $\Theta$ (Deg.)')
    plt.show()

def plot_displacer_thickness(contrasts, displacer_thickness, major_radius):

    pp, ll = np.meshgrid(displacer_thickness, major_radius)

    levels = np.arange(0, 0.43, 0.05)

    plt.figure(1)
    plt.title('Delay plate = 15mm, Displacer Cut angle $\Theta$=45$^{\circ}$')
    im = plt.pcolormesh(ll, pp / 1000, contrasts, shading='gouraud')
    cbar = plt.colorbar(im)
    cbar.set_label('Contrast', rotation=90)
    CS = plt.contour(ll, pp * 1000, contrasts, colors='black', levels=levels)
    plt.clabel(CS, inline=1)
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Displacer plate thickness [mm]')
    plt.show()

    # plt.figure()
    # plt.title('Delay plate = 15mm, Displacer Cut angle $\Theta$=45$^{\circ}$')
    # CS = plt.contour(ll, pp / 1000, contrasts, levels=levels)
    # plt.clabel(CS, inline=1, fontsize=10)
    # cbar2 = plt.colorbar()
    # plt.xlabel('Major Radius [m]')
    # plt.ylabel('Displacer plate thickness [mm]')
    # cbar2.set_label('Contrast', rotation=90)
    # plt.show()

    return

#Get the msesim run output you want, remember to restore the idl file in calculate_1d_contrast.py

major_radius, stokes_full, wavelength_vector, linearly_polarised = get_stokes_components()

# Basically switch which parameter you want to vary by making an array of variables.
# Then go into the function find_optimal_crystal in calculate_1d_contrast.py and swap out the loop for the other variable.
# Pretty clunky but saves the same piece of code being generated 4 separate times...

delay_thickness = 15*10**-3 #np.arange(5*10**-3,26*10**-3,1*10**-3)
displacer_thickness = 3*10**-3 #np.arange(0.5*10**-3, 6*10**-3, 1*10**-3)
displacer_cut_angle = np.arange(0,90,1)*(np.pi/180)
delay_cut_angle = 0.

contrasts, total_polarised, phi_total = find_optimal_crystal(wavelength_vector, stokes_full, delay_thickness=delay_thickness, delay_cut_angle=delay_cut_angle, displacer_thickness=displacer_thickness, displacer_cut_angle=displacer_cut_angle)
#plot_contrast_contourplot(contrasts, delay_thickness, major_radius)
plot_contrast_cut_angle(contrasts, displacer_cut_angle, major_radius)
