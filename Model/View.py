import numpy as np

class CollectionOptics(object):

    def __init__(self):
        self.xyz = [-0,949, -2.228, 0.000] # xyz co-ordinates of collection lens in machine co-ordinates

        self.ck = [-0.750, -0.662, 0.000] #xyz co-ordinates of the collection lens vector (k)
        self.ck = self.ck/np.sqrt(self.ck[0]**2 + self.ck[1]**2 + self.ck[2]**2) #normalise co-ords
        self.cl = np.cross([0.,0.,1.], self.ck) #find horizontal vector l = ZxK/|ZxK|

        self.diameter = 38*10**-3 # diameter of collection lens (m)
        self.efl = 72.98*10**-3 # effective focal length of collection lens (m)


class Camera(object):

    def __init__(self):
        self.pixel_size = 10*10**-6
        self.pixel_number = [1280, 1024] #pixel number in horizontal and vertical direction
        self.x_length = self.pixel_size*self.pixel_number[0]
        self.y_length = self.pixel_size*self.pixel_number[1]