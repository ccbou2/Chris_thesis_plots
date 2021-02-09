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
plt.rc('text', usetex=False)

#-----------------------------------------------------------------------------------------#
# Import data
#-----------------------------------------------------------------------------------------#
# Load lyse dataset
df = data()

# Locate desired shots via last sequence or timestamp
# subdf = df.sequences(last=1) #.iloc[:-2]
# subdf = df.loc['20190925T161248']
try:
	# no faraday beam
	subdf1 = df.loc[['20190923T105432','20190923T111214']]
except KeyError:
	print ('Cant locate sequences 20190923T105432, 20190923T111214 in lyse.')
try:
	# 781.5nm probe
	subdf2 = df.loc[['20190913T134246','20190913T135226']]
except KeyError:
	print ('Cant locate sequences 20190913T134246, 20190913T135226 in lyse.')
try:
	# 790nm probe
	subdf3 = df.loc['20190923T102811']
except KeyError:
	print ('Cant locate sequence 20190923T102811 in lyse.')

# Re-index subdf based on variables we care about (amplitude and frequency for now)
subdf1 = subdf1.sort_values(by = 'faraday_hold_time')
subdf2 = subdf2.sort_values(by = 'faraday_hold_time')
subdf3 = subdf3.sort_values(by = 'faraday_hold_time')

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]

x_col = ('faraday_hold_time')
y_col = ('side','absorption','OD','Fit_fit_number')
# u_y_col = ('side','absorption','OD','u_Fit_fit_number')

x1 = subdf1[x_col][1:]
y1 = subdf1[y_col][1:]
# u_y1 = subdf1[u_y_col][1:]
x2 = subdf2[x_col]
y2 = subdf2[y_col]
x3 = subdf3[x_col]
y3 = subdf3[y_col]

x1, y1 = fit_filter_nans(x1, y1)
x2, y2 = fit_filter_nans(x2, y2)
x3, y3 = fit_filter_nans(x3, y3)

x2_slosh = x2[:4]
y2_slosh = y2[:4]
x2 = x2[4:]
y2 = y2[4:]

#----------------------------------------------------------------------------------#
# Fitting
#----------------------------------------------------------------------------------#

# Exponential decays:
# Model functions
def burnoff_fit(t, N_0, t_0, const):
	return N_0 * np.exp(- t / t_0) + const

def burnoff_fit_off(t, N_0, t_0, t_off, const):
	return N_0 * np.exp(- (t - t_off) / t_0) + const

def linFunc(t, m, c):
	return m*t + c

# Prepare models and params:
# model1 = lmfit.Model(linFunc)
# params1 = model1.make_params(m = 4e5/20, c = 8e5)
model1 = lmfit.Model(burnoff_fit)
params1 = model1.make_params(N_0 = 600e2, t_0 = 20, const = 0)
params1['const'].vary = False
model2 = lmfit.Model(burnoff_fit_off)
params2 = model2.make_params(N_0 = 600e2, t_0 = 0.1, t_off = 0.1, const = 0)
# params2['const'].vary = False
model3 = lmfit.Model(burnoff_fit)
params3 = model3.make_params(N_0 = 600e3, t_0 = 1.5, const = 0)
# params3['const'].vary = False

# Perform fits:
print('Performing fits...')
result1 = model1.fit(y1, params1, t = x1)
result2 = model2.fit(y2, params2, t = x2)
result3 = model3.fit(y3, params3, t = x3)

# Retrieve time constants:
r1_tstring = r'$\tau_0$ = {:.0f}({:.0f})s'.format(result1.params['t_0'].value, result1.params['t_0'].stderr)
r2_tstring = r'$\tau_0$ = {:.3f}({:.0f})s'.format(result2.params['t_0'].value, result2.params['t_0'].stderr*1e3)
r3_tstring = r'$\tau_0$ = {:.2f}({:.0f})s'.format(result3.params['t_0'].value, result3.params['t_0'].stderr*1e2)

print(r1_tstring)
print(r2_tstring)
print(r3_tstring)

#----------------------------------------------------------------------------------#
# plotting
#----------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

fig, ax = plt.subplots(figsize = (10,5))
ax.plot(x1, y1, marker = 'o', linestyle = 'None', label = 'Trap decay')
ax.plot(x2, y2, marker = 'o', linestyle = 'None', label = '781.5nm scattering')
ax.plot(x2_slosh, y2_slosh, marker = 'o', c = 'midnightblue', linestyle = 'None', label = '781.5nm sloshing')
ax.plot(x3, y3, marker = 'o', linestyle = 'None', label = '790nm scattering')
ax.plot(x1, result1.best_fit, marker = 'None', linestyle = '--', label = 'Trap decay fit')
ax.plot(x2, result2.best_fit, marker = 'None', linestyle = '--', label = '781.5nm fit')
ax.plot(x3[1:], result3.best_fit[1:], marker = 'None', linestyle = '--', c = 'm', label = '790nm fit')
ax.set_xscale('log')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,3))
ax.set_ylim(0,9.2e5)
ax.legend()
ax.grid()
# plt.xlim([-0.1,5])
ax.set_xlabel('Experiment time [s]')
ax.set_ylabel('Atom count')
ax.set_title('Loss mechanisms: decay rate comparison')

# Save figures
plt.savefig(os.path.join(folder, 'decayRates.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'decayRates.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))

fig, ax = plt.subplots(figsize = (10,5))
ax.plot(x1, y1, marker = 'o', linestyle = 'None', label = 'Trap decay')
ax.plot(x2, y2, marker = 'o', linestyle = 'None', label = '781.5nm scattering')
ax.plot(x2_slosh, y2_slosh, marker = 'o', c = 'midnightblue', linestyle = 'None', label = '781.5nm sloshing')
ax.plot(x3, y3, marker = 'o', linestyle = 'None', label = '790nm scattering')
# ax.plot(x1, result1.best_fit, marker = 'None', linestyle = '--', label = 'Trap decay fit')
# ax.plot(x2, result2.best_fit, marker = 'None', linestyle = '--', label = '781.5nm fit')
# ax.plot(x3, result3.best_fit, marker = 'None', linestyle = '--', c = 'm', label = '790nm fit')
ax.plot(x1, result1.best_fit, marker = 'None', linestyle = '--')
ax.plot(x2, result2.best_fit, marker = 'None', linestyle = '--')
ax.plot(x3, result3.best_fit, marker = 'None', c = 'm', linestyle = '--')
ax.set_yscale('log')
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,3))
ax.set_xlim(0,5)
ax.legend()
ax.grid()
# plt.xlim([-0.1,5])
ax.set_xlabel('Experiment time [s]')
ax.set_ylabel('Atom count')
ax.set_title('Loss mechanisms: decay rate comparison')

# Annotate with time constants from fits:
ax.text(3.2, 5.4e5, r1_tstring, color = colours[3])
ax.text(0.34, 5e4, r2_tstring, color = colours[4])
ax.text(1.35, 2e5, r3_tstring, color = 'm')

# Save figures
plt.savefig(os.path.join(folder, 'decayRates2.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'decayRates2.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))

plt.show()