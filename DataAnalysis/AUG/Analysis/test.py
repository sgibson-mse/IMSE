
import numpy as np
import matplotlib.pyplot as plt
import animatplot as amp


x = np.arange(0,1760,1)
y = np.arange(0,1960,1)
t = np.linspace(0, 2*np.pi, 30)

X, Y, T = np.meshgrid(x, y, t)

print(X.shape, Y.shape, T.shape)

pcolormesh_data = np.ones((np.shape(X)))
line_data       = pcolormesh_data[20,:,:] # the slice where y=0

# standard matplotlib stuff
# create the different plotting axes
fig, ax = plt.subplots()

ax.set_ylabel('y', labelpad=-5)

fig.suptitle('Multiple blocks')
ax.set_title('Polarisation angle')

# animatplot stuff
# now we make our blocks
pcolormesh_block = amp.blocks.Pcolormesh(X[:,:,0], Y[:,:,0], pcolormesh_data,
                                         ax=ax, t_axis=2, vmin=-1, vmax=1)
plt.colorbar(pcolormesh_block.quad)
timeline = amp.Timeline(t, fps=200)

# now to contruct the animation
anim = amp.Animation([pcolormesh_block], timeline)
anim.controls()

# anim.save_gif('images/multiblock')
plt.show()