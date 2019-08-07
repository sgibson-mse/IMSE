
class Calibrate:

    def __init__(self):

        self.faraday = None #contribution to polarisation angle due to faraday rotation
        self.absolute = None #absolute value due to alignment, etc.
        self.relative = None #channel to channel offset
        self.non_linearity = None #correction due to non linearity - in space and angle. Do the input polarisation vs output polarisation fit like at ANU
