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

# Add Latex physics package to preamble for matplotlib
plt.rc('text', usetex=False)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{physics} \usepackage{amssymb}')

#-----------------------------------------------------------------------------------------#
# Import data
#-----------------------------------------------------------------------------------------#
# Load lyse dataset
df = data()

# Locate desired shots via last sequence or timestamp
# subdf = df.sequences(last=1) #.iloc[:-2]
subdf = df.loc['20190925T162445']
subdf2 = df.loc['20190925T165817']
# subdf1 = df.loc[['20191001T144113','20191001T145036','20191001T145244']]
# subdf2 = df.loc['20191017T141450']
# seq_str = subdf.sequence_string()
title_str = 'Second Rabi spectroscopy - Periodogram moments'
title_str2 = 'Second Rabi spectroscopy - Sinusoidal fits'

# # Re-index subdf based on variables we care about (amplitude and frequency for now)
# subdf1.set_index(df.tuplify('magnetom_ac_frequency'), inplace = True)
# subdf2.set_index(df.tuplify('sp_pulse_amp'), inplace = True)
# subdf = subdf.sort_values(by = 'magnetom_ac_frequency')
# subdf2 = subdf2.sort_values(by = 'sp_pulse_amp')

# Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]
# freqdf = subdf.loc[(slice(None), 11000.0), :]

#-----------------------------------------------------------------------------------------#
# Perform calibration fits
#-----------------------------------------------------------------------------------------#

x_col = ('magnetom_ac_frequency')
y_col = ('faraday_demod','Q_fft_centroid')
u_y_col = ('faraday_demod','Q_fft_variance')
y_col2 = ('faraday_demod','fit_rabi2')
u_y_col2 = ('faraday_demod','u_fit_rabi2')

# Import data from subdf for original spectroscopy run
x = subdf[x_col]
y = subdf[y_col]
u_y = subdf[u_y_col]

# Import data from subdf for original spectroscopy run
x2 = subdf2[x_col]
y2 = subdf2[y_col]
u_y2 = subdf2[u_y_col]

# Import data from subdf for original spectroscopy run, for fitted rabi
y12 = subdf[y_col2]
u_y12 = subdf[u_y_col2]

# Import data from subdf for original spectroscopy run, for fitted rabi
y22 = subdf2[y_col2]
u_y22 = subdf2[u_y_col2]

# #----------------------------------------------------------------------------------#
# # Fitting
# #----------------------------------------------------------------------------------#

# # Fitting via lmfit
# #----------------------------------------------------------------------------------#
# def linFunc(x, m, c):
# 	return m*x + c

# def powerLaw(x, m, q, c):
# 	return m*x**q + c

# def quadLinFunc(x, ml, mq, c):
# 	return ml*x + mq*x**2 + c

# def cubicFunc(x, ml, mq, mc, c):
# 	return ml*x + mq*x**2 + mc*x**3 + c

# # Define model and set parameter guesses
# model = lmfit.Model(linFunc)
# params = model.make_params(m = 2000, c = 100)
# model2 = lmfit.Model(powerLaw)
# params2 = model2.make_params(m = 2000, q = 1, c = 100)
# # model3 = lmfit.Model(quadLinFunc)
# # params3 = model3.make_params(ml = 2000, mq = 1, c = 100)
# model4 = lmfit.Model(cubicFunc)
# params4 = model4.make_params(ml = 2000, mq = 1, mc = 0.5, c = 100)

# # Perform fit
# print('Performing fit via linear function...')
# # result = model.fit(y1, params, x=x1)
# result = model.fit(y2, params, x=x2)
# print('Performing fit via power law function...')
# # result2 = model2.fit(y1, params2, x=x1)
# result2 = model2.fit(y2, params2, x=x2)
# # print('Performing fit to quadratic plus linear function...')
# # result3 = model3.fit(y0.values, params3, x=x.values)
# print('Performing fit to cubic polynomial function...')
# result4 = model4.fit(y2, params4, x=x2)

# # # Print fit report
# # print(result.fit_report())
# # print(result2.fit_report())
# # # print(result3.fit_report())
# # print(result4.fit_report())

#----------------------------------------------------------------------------------#
# Plots
#----------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

fig = plt.figure(figsize = (10,7))
ax1 = fig.add_subplot(2,1,1)
ax1.errorbar(x, y, u_y, linestyle = 'None', c = 'grey', label = 'u(data)')
ax1.plot(x, y, marker = 'o', linestyle = 'None', label = 'data')
ax1.set_ylabel('Peak frequency $\Omega_2$ [Hz]')
ax1.grid()
ax1.legend()
ax1.set_title(title_str)

ax2 = fig.add_subplot(2,1,2)
ax2.errorbar(x2, y2, u_y2, linestyle = 'None', c = 'midnightblue', label = 'u(data)')
ax2.plot(x2, y2, marker = 'o', c = colours[1], linestyle = 'None', label = 'data')
ax2.set_xlabel('Signal frequency $f_s$ [kHz]')
ax2.set_ylabel('Peak frequency $\Omega_2$ [Hz]')
ax2.grid()
ax2.legend()
mark_inset(ax1, ax2, loc1=1, loc2=2, fc="crimson", alpha = 0.2, ec='crimson', ls='dashdot', lw = 1)

# Save figures
plt.savefig(os.path.join(folder, 'rabi_spectroscopy.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'rabi_spectroscopy.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))


# Now plot for fitted rabi, rather than periodogram moments
fig2 = plt.figure(figsize = (10,7))
ax21 = fig2.add_subplot(2,1,1)
ax21.errorbar(x, y12, u_y12*10, linestyle = 'None', c = 'grey', label = r'u(data) $\times 10$')
ax21.plot(x, y12, marker = 'o', linestyle = 'None', label = 'data')
ax21.set_ylabel('Fit $\Omega_2$ [Hz]')
ax21.grid()
ax21.legend()
ax21.set_title(title_str2)

ax22 = fig2.add_subplot(2,1,2)
ax22.errorbar(x2, y22, u_y22*10, linestyle = 'None', c = 'midnightblue', label = r'u(data) $\times 10$')
ax22.plot(x2, y22, marker = 'o', c = colours[1], linestyle = 'None', label = 'data')
ax22.set_xlabel('Signal frequency $f_s$ [kHz]')
ax22.set_ylabel('Fit $\Omega_2$ [Hz]')
ax22.grid()
ax22.legend()
mark_inset(ax21, ax22, loc1=1, loc2=2, fc="crimson", alpha = 0.2, ec='crimson', ls='dashdot', lw = 1)

# Save figures
plt.savefig(os.path.join(folder, 'rabi_spectroscopy_fits.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'rabi_spectroscopy_fits.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))

# Now plot for fitted rabi, rather than periodogram moments
fig3 = plt.figure(figsize = (10,5))
ax31 = fig3.add_subplot(1,1,1)
ax31.errorbar(x[10:], y12[10:], u_y12[10:]*10, linestyle = 'None', c = 'grey', label = r'u(200 Hz sweep) $\times 10$')
ax31.plot(x[10:], y12[10:], marker = 'o', linestyle = 'None', label = 'cut 200 Hz sweep')
ax31.errorbar(x2, y22, u_y22*10, linestyle = 'None', c = 'midnightblue', label = r'u(20 Hz sweep) $\times 10$')
ax31.plot(x2, y22, marker = 'o', c = colours[1], linestyle = 'None', label = '20 Hz sweep')
ax31.set_ylabel('Fit $\Omega_2$ [Hz]')
ax31.set_xlabel('Signal frequency $f_s$ [kHz]')
ax31.grid()
ax31.legend()
ax31.set_title(title_str2)

# ax22 = fig2.add_subplot(2,1,2)
# ax22.errorbar(x2, y22, u_y22*10, linestyle = 'None', c = 'midnightblue', label = r'u(data) $\times 10$')
# ax22.plot(x2, y22, marker = 'o', c = colours[1], linestyle = 'None', label = 'data')
# ax22.set_xlabel('Signal frequency $f_s$ [kHz]')
# ax22.set_ylabel('Fit $\Omega_2$ [Hz]')
# ax22.grid()
# ax22.legend()
# mark_inset(ax21, ax22, loc1=1, loc2=2, fc="crimson", alpha = 0.2, ec='crimson', ls='dashdot', lw = 1)

# Save figures
plt.savefig(os.path.join(folder, 'rabi_spectroscopy_fits_cut.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'rabi_spectroscopy_fits_cut.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))

plt.show()