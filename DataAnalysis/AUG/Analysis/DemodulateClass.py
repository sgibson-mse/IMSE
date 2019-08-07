import numpy as np

class Demod:

    def __init__(self, image):
        self.image = image

    def subtract_background(self):
        pass

    def subtract_neutrons(self):
        pass

    def fft(self):
        pass



class Filter:

    def __init__(self, shape, size):
        self.shape = shape
        self.size = size

class Box(Filter):

    def __init__(self, shape, size):
        super().__init__(shape, size)

        self.size_x = None
        self.size_y = None
        self.centre = None

    def make(self):
        box = np.zeros((self.size))
        box[int(self.centre[1] - self.size_y / 2): int(self.centre[1] + self.size_y / 2), int(self.centre[0] - (self.size_x / 2)): int(self.centre[0] + self.size_x / 2)] = 1.
        return box



#demodulation process
#Take image and background subtract
#FFT image
#Find carrier frequencies
#make a filter with a certain size and center and shape
#apply filter and IFFT
#find the amplitude and phase of each of the carrier frequencies
#get polarisation angle