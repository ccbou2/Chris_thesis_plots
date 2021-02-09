#!python3
from __future__ import division
from lyse import *
from pylab import *
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

#-----------------------------------------------------------------------------------------#
# Import data
#-----------------------------------------------------------------------------------------#
# Load lyse dataset
df = data()

# Locate desired shots via last sequence or timestamp
# subdf = df.sequences(last=1) #.iloc[:-2]
# subdf = df.loc['20190925T161248']
try:
	subdf = df.loc['20190913T144416']
except KeyError:
	print ('Cant locate shot 20190913T144416 in lyse.')
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
# subdf.set_index([df.tuplify('magnetom_ac_amplitude'), df.tuplify('magnetom_ac_frequency')], inplace = True)

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]

x_col = ('faraday_hold_time')
# Take X0 and Y0 values from OD fit
y1_col = ('side','absorption','OD', 'Gaussian_X0')
u_y1_col = ('side','absorption','OD', 'u_Gaussian_X0')
y2_col = ('side','absorption','OD', 'Gaussian_Y0')
u_y2_col = ('side','absorption','OD', 'u_Gaussian_Y0')

x = subdf[x_col]
y1 = np.array(subdf[y1_col])
u_y1 = np.array(subdf[y1_col])/1e2
y2 = np.array(subdf[y2_col])
u_y2 = np.array(subdf[y2_col])/1e3

# Compute radius from change in X0 and Y0 positions of OD fit centroid
yr = np.sqrt((y1 - y1[0])**2 + (y2 - y2[0])**2)
u_yr = yr*1e1*np.sqrt((u_y1/y1)**2 + (u_y2/y2)**2)
y1_diff = y1 - y1[0]
y2_diff = y2 - y2[0]
y_col = 'Radial position change (pixels)'

#-----------------------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

plt.figure('dipole_slosh', figsize=(10,6))
# plt.errorbar(x, y1_diff, u_y1, c = colours[0], ls = 'none', label = 'X0 error', alpha = 0.5)
plt.plot(x, y1_diff, c = colours[0], marker = 'o', linestyle = 'dotted', label = 'X0 change')
# plt.errorbar(x, y2_diff, u_y2, c = colours[1], ls = 'none', label = 'Y0 error', alpha = 0.5)
plt.plot(x, y2_diff, c = colours[1], marker = 'o', linestyle = 'dotted', label = 'Y0 change')
# plt.errorbar(x, yr, u_yr, c = colours[2], ls = 'none', label = 'Radial error', alpha = 0.5)
plt.plot(x, yr, c = colours[2], marker = 'o', linestyle = 'dotted', label = 'Radial change')
plt.ylabel('Position change [pixels]')
plt.xlabel('Hold time [s]')
plt.grid()
plt.legend()
plt.title('Dipole trap sloshing, 781.5nm probe')
plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'dipole_trap_slosh.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'dipole_trap_slosh.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))