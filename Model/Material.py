class Material():

    def __init__(self, name):

        self.name = name
        self.sc_ne, self.sc_no = self._sellmeier_coefficients(name)

    def _sellmeier_coefficients(self, name):

        if name == 'alpha_bbo':
            self.sc_ne = [2.3753, 0.01224, 0.01667, 0.01516]
            self.sc_no = [2.7359, 0.01878, 0.01822, 0.01354]

        if self.name == 'lithium_niobate':

            self.sc_ne = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
            self.sc_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]

        return self.sc_ne, self.sc_no

    def refractive_index(self, sellmeier_coefficients, material_name, wavelength):

        """
        :param sellmeier_coefficients: Coefficients required to calculate the refractive indices of the specific crystal material. Taken from A thormans thesis.. there are many others...
        :param name: Crystal type - either alpha_bbo or lithium_niobate
        :return: refractive index of the material
        """

        if material_name == 'alpha_bbo':
            return sellmeier_coefficients[0] + sellmeier_coefficients[1]/((wavelength*10**6)**2 - sellmeier_coefficients[2]) - (sellmeier_coefficients[3]*(wavelength*10**6)**2)

        if material_name == 'lithium_niobate':
            return 1 + (sellmeier_coefficients[0]*(wavelength*10**6)**2)/((wavelength*10**6)**2 - sellmeier_coefficients[1]) + (sellmeier_coefficients[2]*(wavelength*10**6)**2)/((wavelength*10**6)**2 - sellmeier_coefficients[3]) + (sellmeier_coefficients[4]*(wavelength*10**6)**2) / ((wavelength*10**6)**2 - sellmeier_coefficients[5])
        else:
            print('Crystal not yet implemented!')