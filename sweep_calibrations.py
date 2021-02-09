#!python3
from __future__ import division
from lyse import *
from pylab import *
import numpy as np
from analysislib.common import *
from analysislib.spinor.aliases import *
from analysislib.spinor.faraday.faraday_aux import load_stft, plot_stft
from analysislib.spinor.Chris_thesis_plots.func_lib import *
from scipy import signal, special
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import labscript_utils.h5_lock
import h5py
import lmfit
import pandas as pd

# Folder for saving output files
folder = 'C:/Users/ccbou2/GitHub/HonoursThesis/Figures'
plt.rc('text', usetex=False)

#-----------------------------------------------------------------------------------------#
# Import data
#-----------------------------------------------------------------------------------------#
# Load lyse dataset
df = data()

# Locate desired shots via last sequence or timestamp
# subdf = df.sequences(last=1) #.iloc[:-2]
subdf1 = df.loc[['20191001T144113','20191001T145036','20191001T145244']]
subdf2 = df.loc['20191017T141450']
# seq_str = subdf.sequence_string()
title_str = 'Calibration of first Rabi frequency'

# # Re-index subdf based on variables we care about (amplitude and frequency for now)
# subdf1.set_index(df.tuplify('sp_pulse_amp'), inplace = True)
# subdf2.set_index(df.tuplify('sp_pulse_amp'), inplace = True)
subdf1 = subdf1.sort_values(by = 'sp_pulse_amp')
subdf2 = subdf2.sort_values(by = 'sp_pulse_amp')

# Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]
# freqdf = subdf.loc[(slice(None), 11000.0), :]

#-----------------------------------------------------------------------------------------#
# Perform calibration fits
#-----------------------------------------------------------------------------------------#

x_col = ('sp_pulse_amp')
y_col = ('fit_quadratic_sidebands','fRabi')
u_y_col = ('fit_quadratic_sidebands','u_fRabi')

# Import data from subdf1 for original calib
x1 = subdf1[x_col]
y1 = subdf1[y_col]
u_y1 = subdf1[u_y_col]

# Import data from subdf2 for newer calibrations
x2 = subdf2[x_col]
y2 = subdf2[y_col]
u_y2 = subdf2[u_y_col]

# filter lists based on nan entries in y;
tempx = []
tempy = []
tempuy = []
for i in range(len(y1)):
	if not np.isnan(y1[i]):
		tempx.append(x1[i])
		tempy.append(y1[i])
		tempuy.append(u_y1[i])
x1 = np.array(tempx)
y1 = np.array(tempy)/1e3
u_y1 = np.array(tempuy)/1e3

tempx = []
tempy = []
tempuy = []
for i in range(len(y2)):
	if not np.isnan(y2[i]):
		tempx.append(x2[i])
		tempy.append(y2[i])
		tempuy.append(u_y2[i])
x2 = np.array(tempx)
y2 = np.array(tempy)/1e3
u_y2 = np.array(tempuy)/1e3

#----------------------------------------------------------------------------------#
# Fitting
#----------------------------------------------------------------------------------#

# Fitting via lmfit
#----------------------------------------------------------------------------------#
def linFunc(x, m, c):
	return m*x + c

def powerLaw(x, m, q, c):
	return m*x**q + c

def quadLinFunc(x, ml, mq, c):
	return ml*x + mq*x**2 + c

def cubicFunc(x, ml, mq, mc, c):
	return ml*x + mq*x**2 + mc*x**3 + c

# Define model and set parameter guesses
model = lmfit.Model(linFunc)
params = model.make_params(m = 2000, c = 100)
model2 = lmfit.Model(powerLaw)
params2 = model2.make_params(m = 2000, q = 1, c = 100)
# model3 = lmfit.Model(quadLinFunc)
# params3 = model3.make_params(ml = 2000, mq = 1, c = 100)
model4 = lmfit.Model(cubicFunc)
params4 = model4.make_params(ml = 2000, mq = 1, mc = 0.5, c = 100)

# Perform fit
print('Performing fit via linear function...')
# result = model.fit(y1, params, x=x1)
result = model.fit(y2, params, x=x2)
print('Performing fit via power law function...')
# result2 = model2.fit(y1, params2, x=x1)
result2 = model2.fit(y2, params2, x=x2)
# print('Performing fit to quadratic plus linear function...')
# result3 = model3.fit(y0.values, params3, x=x.values)
print('Performing fit to cubic polynomial function...')
result4 = model4.fit(y2, params4, x=x2)

# # Print fit report
# print(result.fit_report())
# print(result2.fit_report())
# # print(result3.fit_report())
# print(result4.fit_report())

#----------------------------------------------------------------------------------#
# Plots
#----------------------------------------------------------------------------------#

# fig = plt.figure(figsize = (16,9))
# # plt.errorbar(x1, y1, u_y1, linestyle = 'None', c = 'k', label = 'u(first dataset)')
# # # plt.plot(x1, y1, 'bo', label = 'first dataset')
# # plt.plot(x1, y1, marker = 'o', linestyle = 'None', label = 'first dataset')
# plt.errorbar(x2, y2, u_y2, linestyle = 'None', c = 'k', label = 'u(second dataset)')
# # plt.plot(x2, y2, 'ro', label = 'second dataset')
# plt.plot(x2, y2, marker = 'o', linestyle = 'None', label = 'second dataset')
# # plt.plot(x1, result.best_fit, 'r--', label = 'linear fit (first)')
# plt.plot(x2, result.best_fit, 'r--', label = 'linear fit (second)')
# # plt.plot(x1, result2.best_fit, 'g--', label = 'power law fit (first)')
# plt.plot(x2, result2.best_fit, 'g--', label = 'power law fit (second)')
# plt.plot(x2, result4.best_fit, 'k--', label = 'cubic fit (second)')
# plt.xlabel('RF amplitude [Vpp]')
# plt.ylabel(r'$\Omega_1$ [kHz]')
# plt.grid()
# plt.legend(loc='best')
# plt.title(title_str)
# plt.show()

fig = plt.figure(figsize = (10,6))
ax = plt.subplot(111)
axins = zoomed_inset_axes(ax, 2.5, loc = 4, borderpad = 0.7)
axins.set_xlim(0.35, 0.5)
axins.set_ylim(5.1, 7.8)
ax.set_xlim(0.14,1.01)
ax.set_ylim(2,17)
# axins.xaxis.set_visible(False)
# axins.yaxis.set_visible(False)
# plt.errorbar(x1, y1, u_y1, linestyle = 'None', c = 'k', label = 'u(first dataset)')
# # plt.plot(x1, y1, 'bo', label = 'first dataset')
# plt.plot(x1, y1, marker = 'o', linestyle = 'None', label = 'first dataset')
ax.errorbar(x2, y2, u_y2, linestyle = 'None', c = 'grey', label = 'u(data)')
axins.errorbar(x2, y2, u_y2, linestyle = 'None', c = 'grey')
# plt.plot(x2, y2, 'ro', label = 'second dataset')
ax.plot(x2, y2, marker = 'o', linestyle = 'None', label = 'data')
axins.plot(x2, y2, marker = 'o', linestyle = 'None')
# plt.plot(x1, result.best_fit, 'r--', label = 'linear fit (first)')
ax.plot(x2, result.best_fit, 'r--', label = 'linear fit')
axins.plot(x2, result.best_fit, 'r--')
# plt.plot(x1, result2.best_fit, 'g--', label = 'power law fit (first)')
ax.plot(x2, result2.best_fit, 'g--', label = 'power law fit')
axins.plot(x2, result2.best_fit, 'g--')
ax.plot(x2, result4.best_fit, 'k--', label = 'cubic fit')
axins.plot(x2, result4.best_fit, 'k--')
ax.set_xlabel('RF amplitude [Vpp]')
ax.set_ylabel(r'$\Omega_{1R}$ [kHz]')
ax.grid()
axins.grid()
axins.axes.get_xaxis().set_ticklabels([])
# axins.axes.get_yaxis().set_ticklabels([])
# ax.legend(loc='lower right')
ax.legend(loc='upper left')
ax.set_title(title_str)
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec='yellowgreen', ls='dashdot', lw = 2)
# mark_inset(ax, axins, loc1=2, loc2=3, fc="slategrey", alpha = 0.2, ec='k', ls='dashdot', lw = 2)
mark_inset(ax, axins, loc1=2, loc2=3, fc="midnightblue", alpha = 0.2, ec='midnightblue', ls='dashdot', lw = 1)
plt.show()
# Save figures
plt.savefig(os.path.join(folder, 'sweepCalib.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'sweepCalib.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))