import numpy as np

class DipoleMatrix(object):

    def __init__(self):

        self.R31_20 = 3.065
        self.R32_21 = 4.748
        self.R30_21 = 0.9384

        self.Dz = self.along_z()
        self.Dr = self.right_hand()
        self.Dl = self.left_hand()


    def along_z(self):

        Dmz =  np.array([[0, 0, 0, np.sqrt(1/3)*self.R30_21],
        [np.sqrt(1/3)*self.R31_20, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, np.sqrt(4/15)*self.R32_21],
        [0, np.sqrt(1/5)*self.R32_21, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]])

        return Dmz

    def right_hand(self):
        # dipole matrix for the right-handed polarization - equation (2.39) on page 49 in PhD thesis of Howard Yuh (PPPL, 2005)

        #                  2 0 0       ,                 2 1 1       ,           2 1-1       ,                           2 1 0
        Dmr     = np.array([[                    0.,                  0.,             -np.sqrt(2./3.)*self.R30_21,                  0.],                  #  3 0 0
                    [                  0.,                  0.,                  0.,                                      0.],                  #  3 1 0
                    [  np.sqrt(2./3.)*self.R31_20,          0.,                  0.,                                      0.],                  #  3 1 1
                    [                  0.,                  0.,                  0.,                                      0.],                  #  3 1-1
                    [                  0.,                  0.,             np.sqrt(2./15.)*self.R32_21,                  0.],                  #  3 2 0
                    [                  0.,                  0.,                  0.,                                np.sqrt(2./5.)*self.R32_21],#  3 2 1
                    [                  0.,                  0.,                  0.,                                      0.],                  #  3 2-1
                    [                  0.,  np.sqrt(4./5.)*self.R32_21,          0.,                                      0.],                  #  3 2 2
                    [                  0.,                  0.,                  0.,                                      0.] ]  )               # 3 2-2

        return Dmr

    def left_hand(self):
        #                  2 0 0       ,                          2 1 1       ,                  2 1-1       ,          2 1 0
        Dml     =  np.array([[                    0.,                  np.sqrt(2/3)*self.R30_21,             0,                      0.],                  #  3 0 0
                    [                  0.,                  0.,                                   0.,                     0.],                  #  3 1 0
                    [                  0.,                  0.,                                   0.,                     0.],                  #  3 1 1
                    [  -np.sqrt(2./3.)*self.R31_20,         0.,                                   0.,                     0.],                  #  3 1-1
                    [                  0.,            -np.sqrt(2/15)*self.R32_21,                 0,                      0.],                  #  3 2 0
                    [                  0.,                  0.,                                   0.,                     0.],                  #  3 2 1
                    [                  0.,                  0.,                                   0.,            -np.sqrt(2./5.)*self.R32_21],   #  3 2-1
                    [                  0.,                  0.,                                   0.,                     0.],                  #  3 2 2
                    [                  0.,                  0.,                   -np.sqrt(4./5.)*self.R32_21,            0.] ])               # 3 2-2

        return Dml

dp = DipoleMatrix()

Dx = np.transpose(0.5 * (dp.Dl + dp.Dr) * (0 + 1j))
Dy = np.transpose(0.5 * (dp.Dl - dp.Dr) * (0 + 1j))
Dz = np.transpose(dp.Dz * (1+0j))