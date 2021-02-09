#!python3
from __future__ import division
from lyse import *
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
from analysislib.common import *
from analysislib.spinor.aliases import *
from analysislib.spinor.faraday.faraday_aux import load_stft, plot_stft
from analysislib.spinor.Chris_thesis_plots.func_lib import *
from analysislib.spinor.parcyl import parcyd, rhoVec
from scipy import signal, special
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import labscript_utils.h5_lock
import h5py
import lmfit
import pandas as pd

# Folder for saving output files
folder = 'C:/Users/ccbou2/GitHub/HonoursThesis/Figures'

# Add Latex physics package to preamble for matplotlib
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{amssymb}')

t = np.linspace(0,1,200)
Fx = -np.cos(2*np.pi*t)

tdSup = [0.25 - 0.05, 0.25+0.05]
tSup = t[np.logical_and(t > tdSup[0], t < tdSup[1])]
dSup = -np.cos(2*np.pi*tSup)
# dSup = [-np.cos(2*np.pi*0.2), -np.cos(2*np.pi*0.3)]
tdEig = [0.5 - 0.05, 0.5 + 0.05]
tEig = t[np.logical_and(t > tdEig[0], t < tdEig[1])]
dEig = -np.cos(2*np.pi*tEig)
# dEig = [-np.cos(2*np.pi*0.45),-np.cos(2*np.pi*0.55)]

# gradAnti = 2*np.pi*np.sin(2*np.pi*0)
# gradNode = 2*np.pi*np.sin(2*np.pi*0.25)

# tAnti = t[t < 0.15]
# tNode = t[np.logical_and(t > 0.1, t < 0.4)]

# lineAnti = gradAnti*(tAnti - 0) - np.cos(2*np.pi*0)
# lineNode = gradNode*(tNode - 0.25) - np.cos(2*np.pi*0.25)

# Plot sine wave:
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(1,1,1)
ax.plot(t, Fx, label = 'State projection')
ax.plot(tEig, dEig, label = '$\delta\expval{\hat{F}_z}$ eigenstate', ls = '--', lw = 3)
ax.scatter([0.5], [1], marker = 'o', c = 'k', label = 'eigenstate')
# ax.scatter(tdEig, dEig, marker = '|', c = 'k')
ax.plot(tSup, dSup, label = '$\delta\expval{\hat{F}_z}$ superposition', c = 'crimson', ls = '--', lw = 3)
ax.scatter([0.25], [0], marker = 'o', c = 'midnightblue', label = 'superposition')
# ax.scatter(tdSup, dSup, marker = '|', c = 'slategrey')
# ax.plot(tAnti, lineAnti, ls = '-', alpha = 0.75)
# ax.plot(tNode, lineNode, ls = '-', alpha = 0.75)
ax.set_xlabel(r'\text{Time/Period}')
ax.set_ylabel(r'$\expval{\hat{F}_z}$')
ax.set_title(r'\text{Sensitivity to projection changes}')
ax.set_xlim(0,1)
ax.set_ylim(-1.1,1.1)
ax.grid()
ax.legend()

plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'projection_sensitivity.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'projection_sensitivity.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))