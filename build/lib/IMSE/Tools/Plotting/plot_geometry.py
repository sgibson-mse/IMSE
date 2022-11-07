import numpy as np
import matplotlib.pyplot as plt



def plot_mast(r_in, r_out, r_mid):

    theta = np.arange(0,360,1)*np.pi/180.
    r_space = np.arange(0,3,1)

    plt.figure()

    #Plot the machine
    plt.plot(r_in*np.cos(theta), r_in*np.sin(theta), color='gray')
    plt.plot(r_out*np.cos(theta), r_out*np.sin(theta), color='gray')
    plt.plot(r_mid*np.cos(theta), r_mid*np.sin(theta), '--', color='C2')
    plt.show()

    return

r_in = 0.5
r_mid = 1
r_out = 1.5
plot_mast(r_in, r_out, r_mid)