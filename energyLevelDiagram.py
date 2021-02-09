#!python3

#!python3
from __future__ import division
from lyse import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
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

# Add Latex physics package to preamble for matplotlib
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{amssymb}')

# Folder for saving output files
folder = 'C:/Users/ccbou2/GitHub/HonoursThesis/Figures'

detunings = np.linspace(-15000, 15000, 1000)
plusState = 0.1 * detunings
minusState = -0.1 * detunings
plus2State = np.sqrt(300**2 + plusState**2)
minus2State = -np.sqrt(300**2 + minusState**2)
horizontalAxis = np.zeros(len(detunings))

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(1,1,1)
ax.plot(detunings, plusState, c = 'slategrey', ls = '--')
ax.plot(detunings, minusState, c = 'slategrey', ls = '--')
ax.plot(detunings, plus2State)
ax.plot(detunings, minus2State)
# ax.plot(detunings, horizontalAxis, c = 'k', ls = '-')
# plt.axvline(0, ymin=np.min(minus2State), ymax=np.max(plus2State), c='k', ls='-')
ax.arrow(0, -1500, 0, 3000, length_includes_head=  True, fc = 'k', ec = 'k', head_width = 400, head_length = 75, alpha = 0.7)
ax.arrow(-15000, 0, 30000, 0, length_includes_head=  True, fc = 'k', ec = 'k', head_width = 60, head_length = 500, alpha = 0.7)
ax.set_xlim(-15000, 15000)
ax.set_ylim(-1500, 1500)

# Axes labels
ax.text(-250,1550,'E', size = 'x-large')
ax.text(15500,-50,'$\\Delta$', size=  'x-large')

# State labels
ax.text(-14000,1100,'$\\ket{+_{\\text{z}}}$', size = 'x-large', color = 'grey')
ax.text(13000,-1200,'$\\ket{+_{\\text{z}}}$', size = 'x-large', color = 'grey')
ax.text(-14000,-1200,'$\\ket{-_{\\text{z}}}$', size = 'x-large', color = 'grey')
ax.text(13000,1100,'$\\ket{-_{\\text{z}}}$', size = 'x-large', color = 'grey')
ax.text(-3000,500,'$\\ket{+_{\\text{i}}}$', size = 'x-large', color = colours[0])
ax.text(1800,-600,'$\\ket{-_{\\text{i}}}$', size = 'x-large', color = colours[1])

# Avoided crossing label
ax.text(-8000,100,'$\\hbar\\Omega_1$', size = 'x-large', color = colours[2])
ax.axhline(300, xmin = 0.3, xmax = 0.5, linestyle = 'dashdot', color = colours[2])
ax.axhline(-300, xmin = 0.3, xmax = 0.5, linestyle = 'dashdot', color = colours[2])
ax.arrow(-6000, 0, 0, 295, length_includes_head=  True, fc = colours[2], ec = colours[2], head_width = 400, head_length = 75, alpha = 0.7)
ax.arrow(-6000, 0, 0, -295, length_includes_head=  True, fc = colours[2], ec = colours[2], head_width = 400, head_length = 75, alpha = 0.7)

# Remove frame
frame = plt.gca()
frame.axis('off')
# xlabel(r'$\Delta$')
# h = ylabel('E')
# h.set_rotation(0)
# grid()
plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'dressedLevelsLab.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'dressedLevelsLab.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))