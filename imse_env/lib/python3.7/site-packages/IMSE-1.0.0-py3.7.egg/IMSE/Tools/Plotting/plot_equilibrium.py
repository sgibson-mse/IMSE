from scipy.io import readsav
import numpy as np
import matplotlib.pyplot as plt

data = readsav('/work/sgibson/msesim/equi/equi_edgecurrent_MASTU.sav')

plt.figure()
plt.imshow(data['fluxcoord'])
plt.colorbar()
plt.show()