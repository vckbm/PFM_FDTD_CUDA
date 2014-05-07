#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from os import sys

X, Y, Z = np.loadtxt('incr_'+sys.argv[1]+'.txt', unpack=True, usecols=[0,1,3])

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

bar = np.linspace(0.4, 0.5, 128)

plt.tricontourf(X, Y, Z, bar, cmap=plt.cm.jet, extend='both')
plt.colorbar()

fig.savefig(sys.argv[1]+'.png', format='png')
fig.clear()
#plt.show()
