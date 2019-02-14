import numpy as np

class Lens():

    def __init__(self, focal_length):

        self.transmission = 0.9
        self.focal_length = focal_length

class Polarizer():

    def __init__(self, theta):

        self.theta = theta*(np.pi/180.)
        self.transmission = 0.6

    def get_phase(self, *args):
        return self.theta

    def matrix(self, theta):

        return np.array([[1, 0, 0, 0],
                        [0, np.cos(2*theta), np.sin(2*theta), 0],
                        [0, -1*np.sin(2*theta), np.cos(2*theta), 0],
                        [0, 0, 0, 1]])

class Filter():

    def __init__(self, fwhm, cwl):

        self.fwhm = fwhm
        self.cwl = cwl
        self.transmission = 0.6
        #transmission function should be calculated here

#Probs add some more optics in here
