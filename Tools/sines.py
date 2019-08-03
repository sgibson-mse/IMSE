import numpy as np

import matplotlib.pyplot as plt

# Get x values of the sine wave

x = np.linspace(0,100,1000)

phase = 1
w = np.pi/2
dw = np.pi/5

w2 = np.pi/3

# Amplitude of the sine wave is sine of a variable like time

amplitudes = 1 + np.cos(2*(w+dw))*np.cos(x)
amplitudep = 1 + np.cos(2*w)*np.cos(x)

amplitudet = 1 + np.cos(2*w2)*np.cos(x)

sum = 0.5*(amplitudes+amplitudep)

# plt a sine wave using time and amplitude obtained for the sine wave

plt.plot(x, amplitudes, label='$\sigma$')
plt.plot(x, amplitudep, label='$\pi$')

plt.plot(x, amplitudet, label='phase shift')


# Give y axis label for the sine wave plt

plt.ylabel('I$_{O}$ Amplitude')
plt.legend()
plt.axhline(y=1, color='k')

plt.show()