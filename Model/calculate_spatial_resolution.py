from Model.load_msesim_output import load_msesim_spectrum
import numpy as np
import matplotlib.pyplot as plt

data = load_msesim_spectrum()

r_center = data['resolution_vector(R)'][:,0].reshape(21,21)

Rc = r_center[10,:]

beam_width = 0.15 #m at the beam port
beam_divergence = 0.7 * (np.pi/180.)
focus = [14, 5.17] #horizontal/verical focus
beam_duct = [0.539, -1.926, 0.0] #port in xyz
beam_source = np.array([0.188, -6.88, 0.])  # machine coords of beam source

# vector between source and duct
source_to_duct = beam_duct - beam_source

beam_slope = (beam_duct[1] - beam_source[1]) / (beam_duct[0] - beam_source[0])
beam_intercept = beam_duct[1] - beam_slope*beam_duct[0]

#optic axis
collection_optics_xyz = np.array([-0.949, -2.228, 0.000])  # machine coords of collection lens

optic_axis = data['optical_axis']
beam_coords = data['central_coordinates']

emission_vectors = (beam_coords - collection_optics_xyz).reshape(21,21,3)
emission_1d = emission_vectors[10,:,:]

#take one of the emission points
emission_point = emission_1d[0,:]

#emission vector lines, slope, intercepts
# m = y2 - y1 / x2 - x1, c = -mx + y

emission_slope = (emission_point[1] - collection_optics_xyz[1]) / (emission_point[0] - collection_optics_xyz[0])
emission_intercept = emission_point[1] - emission_slope*emission_point[0]

# m1 * x + b1 = m2 * x + b2
# m1 * x - m2 * x = b2 - b1

intercept_x = (emission_point[1] - emission_point[0]) / (beam_slope - emission_slope)
intercept_y = beam_slope * intercept_x + emission_point[0]

Rn = np.sqrt(intercept_x**2 + intercept_y**2)
print((Rn - Rc[-1])*(21/1024))

z_vector = [0,0,1]

# normalise it to get the beam unit vector
beam_axis = source_to_duct / (np.sqrt(source_to_duct[0] ** 2 + source_to_duct[1] ** 2 + source_to_duct[2] ** 2))
orientation = np.cross(beam_axis, z_vector)












