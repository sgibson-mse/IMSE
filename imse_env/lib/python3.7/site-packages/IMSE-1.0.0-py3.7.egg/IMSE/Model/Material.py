import numpy as np

class Material():

    def __init__(self, name):

        self.name = name
        self.sc_ne, self.sc_no = self.sellmeier_coefficients(name)
        self.transmission = 0.9

    def sellmeier_coefficients(self, name):

        if name == 'alpha_bbo':
            self.sc_ne = [2.3753, 0.01224, 0.01667, 0.01516]
            self.sc_no = [2.7359, 0.01878, 0.01822, 0.01354]

        if self.name == 'lithium_niobate':

            self.sc_ne = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
            self.sc_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]

        return self.sc_ne, self.sc_no
