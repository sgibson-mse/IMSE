import numpy as np

#Read a .dat binary file which contains images from the IMSE diagnostic system.
import matplotlib.pyplot as plt

def load_binary(filename, FLC):

    """
    :param filename: Binary filename. Binary file contains a header with 3 16 bit unassigned integers denoting number of pixels in x and y and the number of frames taken.
    :param FLC: If the FLC is on, FLC must be true, as there will be double the number of frames (one for each polarisation state)
    :return: Array containing two images for each rotation.
    """

    n_headers = 3 #nx, ny, and n_frames

    with open(filename,'rb') as f:

        header = np.fromfile(f, dtype=np.int16, count=2)

        nx = header[0] #.astype(np.uint32) #change these to 32 bit assigned integers
        ny = header[1]#.astype(np.uint32)

        header_32bit = np.fromfile(f, dtype=np.int32, count=2)

        n_frames = header_32bit[0]
        navg = header_32bit[1]

        header_floats = np.fromfile(f, dtype=np.float32, count=2)

        step = header_floats[0]
        theta = header_floats[1]

        header_32bit_final = np.fromfile(f, dtype=np.int32, count=1)

        status = header_32bit_final[0]

        if FLC:
            #If the FLC is present, there will be 2 frames corresponding to two separate polarisation states.
            n_frames = n_frames*2
        else:
            pass

        data = np.fromfile(f, dtype=np.int32)

        #Order must be fortran-like to preserve the memory layout

        images = data.reshape(nx, ny, n_frames, order='F')

    return images