import numpy as np
from scipy.io import readsav
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sympy import Symbol

lens = np.array([-0.949, -2.228, 0])
k = np.array([0.75, 0.662, 0])
k = k/np.sqrt(np.dot(k,k))
b_duct = (0.539, -1.926, 0)
b_vec = (np.cos(89.1*np.pi/180.), np.sin(89.1*np.pi/180), 0)

t = np.linspace(0,2*np.pi,100)
R = np.linspace(0,2,10)

circle_x = np.cos(t)
circle_y = np.sin(t)

plt.figure()
plt.plot(circle_x, circle_y)
plt.plot(0.2*circle_x, 0.2*circle_y)
plt.show()

#Find the intersection between the collection lens and the beam vector in 3D)
#
#lens_x + a*k_x = duct_x + b*v_x
#lens_y + a*k_y = duct_y + b*v_y
#lens_z + a*k_z = h + b*v_z

#Note: generalised the height term of the beam because the sensor is 2D, so takes into account vertical height of sensor

intersection_xyz = lens - ((b_duct[1]*b_vec[0]-lens[1]*b_vec[0]-b_duct[0]*b_vec[1]+lens[0]*b_vec[1])/(-k[1]*b_vec[0]+k[0]*b_vec[1]))*k

#Convert intersection into R, Phi, Z co-ordinates
int_R = np.sqrt(intersection_xyz[0]**2 + intersection_xyz[1]**2)
int_phi = np.arctan(intersection_xyz[1]/intersection_xyz[0])
int_z = k[2]*(b_duct[1]*b_vec[0] - lens[1]*b_vec[0] - b_duct[0]*b_vec[1] + lens[0]*b_vec[1])/(k[1]*b_vec[0]-k[0]*b_vec[1])

#Convert beam vector to cylindrical co-ordinates
vcyl = (b_vec[0]*np.cos(int_phi) + b_vec[1]*np.sin(int_phi), b_vec[1]*np.cos(int_phi) - b_vec[0]*np.sin(int_phi), 0)
#convert lens vector to cylindrical co-ordinates
kcyl =  (k[0]*np.cos(int_phi) + k[1]*np.sin(int_phi), k[1]*np.cos(int_phi) - k[0]*np.sin(int_phi), k[2])

B = (Symbol('Br',  zero=True),Symbol('Bphi', positive=True), Symbol('Bz', zero=True))

#Get the E field vector
Efld = np.cross(vcyl, B)

#Get the horizontal and vertical vectors in the collection lens
H = np.cross(kcyl, Efld)

V = np.cross(H, kcyl)

coeff1 = np.dot(H,Efld)
coeff2 = np.dot(V,Efld)

print(coeff1)
print(coeff2)
