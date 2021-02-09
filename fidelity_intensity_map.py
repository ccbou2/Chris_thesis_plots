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
from scipy import signal, special
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
subdf = df.loc['20191007T164123']
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
subdf.set_index([df.tuplify('magnetom_ac_amplitude'), df.tuplify('magnetom_ac_frequency')], inplace = True)

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 9000.0), :]

#-----------------------------------------------------------------------------------------#
# Import and process lockin traces
#-----------------------------------------------------------------------------------------#

subRange = 0.35
fidelities = []
amps = []
freqs = []

# Create a list of the paths for each shot in ampdf, and return the filtered Q channel for each
for ind, row in subdf.iterrows():
	# print(freq, row['filepath'])
	path = row['filepath'][0]
	shot_id = os.path.split(path)[-1].replace('.h5', '')

	amps.append(ind[0])
	freqs.append(ind[1])

	# import lockin trace for each
	# print(row['filepath'][0])
	# t_lockin, lockin_Q, lockin_globs = get_lockin_trace(path)

	# retrieve rest of globals
	globs = import_globs(path)

	# Define relevant params from lockin_globs
	# lockin_sample_rate = lockin_globs['actual capture sample rate (Hz)']
	# filter_cutoff = 1e3
	t0 = globs['hobbs_settle'] + globs['magnetom_ac_wait']
	T = globs['magnetom_rabi_sweep_duration']
	tsweep = (t0, t0 + T)
	fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

	# Signal params
	ac_amp = globs['magnetom_ac_amplitude']
	ac_freq = globs['magnetom_ac_frequency']
	fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

	# Compute actual bounds of sweep
	amp_bounds = (amp_calib(fsweep_requested[0], shot_id), amp_calib(fsweep_requested[1], shot_id))
	fsweep_actual = (poly_freq_calib(amp_bounds[0]), poly_freq_calib(amp_bounds[1]))
	rabi_range = abs(fsweep_actual[1] - fsweep_actual[0])

	# # apply butterworth filter (same as rabi_sweep)
	# filter_b, filter_a = signal.butter(8, 2 * filter_cutoff / lockin_sample_rate)
	# filter_Q = signal.filtfilt(filter_b, filter_a, lockin_Q)

	# Repeat Rabi sweep frequency calibrations for lockin trace:
	amp_map = np.vectorize(amp_map)
	# rf_amps_lockin = amp_map(t_lockin, t0, T, amp_bounds[0], amp_bounds[1])
	# rabi_freqs_lockin = poly_freq_calib(rf_amps_lockin)
	# ac_res_time_lockin = t_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]
	# ac_res_rabi_freq = rabi_freqs_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]

	ampToGauss = globs['Bz0_G_per_V']

	def compute_rabi2(amp):
		return 699900*ampToGauss*(amp/1e3)/(np.sqrt(2)*2*np.pi)
	def rabi_to_amp(rabi2):
		return rabi2*1e3*(np.sqrt(2)*2*np.pi)/(699900*ampToGauss)

	# Compute fidelity params 
	rabi2_guess = compute_rabi2(ac_amp)
	sweep_rate = 2*np.pi*rabi_range/(tsweep[1] - tsweep[0])
	det_init = np.abs(fsweep_actual[0] - ac_freq)

	# Define fidelity function
	def compute_fidelity(rabi, rate, det):
		return np.max((np.exp(-2*rate/(np.pi*rabi**2)), 1/(1+(det/rabi)**2)))
	def adiabatic_fidelity(rabi, rate, det):
		return np.exp(-2*rate/(np.pi*rabi**2))
	def asymptotic_fidelity(rabi, rate, det):
		return 1/(1+(det/rabi)**2)

	fid = compute_fidelity(rabi2_guess, sweep_rate, det_init)

	fidelities.append(fid)

	# # Restrict data to desired range for plotting
	# filter_Q_sub = filter_Q[np.logical_and(rabi_freqs_lockin > ac_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_freq + rabi_range*subRange/2)]
	# t_lockin_sub = t_lockin[np.logical_and(rabi_freqs_lockin > ac_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_freq + rabi_range*subRange/2)]

	# Write filter_Q to dataframe, with freq as the row index
	# dataSeries = pd.DataFrame({'amp': amp[0], 't_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq})
	# ampDatadf.append({'t_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq}, name = freq)
	# freqDatadf = freqDatadf.append(dataSeries)

# Generate background intensity map
# ampInts = np.logspace(2, -1, 250)
ampInts = np.linspace(10, 0.1, 500)
# ampInts = np.logspace(-1, 2, 250)
rabiInts = compute_rabi2(ampInts)
# detInts = np.logspace(2, 5, 500)
detInts = np.linspace(100, 20000, 500)
# fidInts = np.zeros((250,250))
fidInts = np.zeros((500,500))
# fidIntsAd = np.zeros((250,250))
fidIntsAd = np.zeros((500,500))
# fidIntsAs = np.zeros((250,250))
fidIntsAs = np.zeros((500,500))

# Compute image
for i in range(len(rabiInts)):
	for j in range(len(detInts)):
		fidInts[i,j] = compute_fidelity(rabiInts[i], sweep_rate, detInts[j])
		fidIntsAd[i,j] = adiabatic_fidelity(rabiInts[i], sweep_rate, detInts[j])
		fidIntsAs[i,j] = asymptotic_fidelity(rabiInts[i], sweep_rate, detInts[j])

# Compute adiabatic, asymptotic and metrology lims to mark on plots
# adiabaticAmpLim = (np.sqrt(2)*2*np.pi/(699900*ampToGauss))*np.sqrt(-2*sweep_rate/(np.pi*np.log(0.03)))
def adiabatic_lim(P_LZ = 0.1):
	return rabi_to_amp(np.sqrt(-2*sweep_rate/(np.pi*np.log(P_LZ))))
# adiabaticAmpLim = (np.sqrt(2)*2*np.pi/(699900*ampToGauss))*np.sqrt(-2*sweep_rate/(np.pi*np.log(0.03)))
# adiabaticAmpLim = adiabatic_lim(0.1)
adiabaticAmpLim = adiabatic_lim(0.05)
def asymptotic_lim(dets, P_LZ = 0.1):
	return rabi_to_amp(np.sqrt(dets**2 * P_LZ/(1 - P_LZ)))
# asymp_lims = asymp_vec_lim(detInts, 0.1)
asymp_lims = asymptotic_lim(detInts, 0.05)
# asymp_lims_03 = asymptotic_lim(detInts, 0.03)
# asymp_lims_1 = asymptotic_lim(detInts, 0.1)
# asymp_fid_lims = asymptotic_fidelity(asymp_lims, sweep_rate, detInts)

# # hacked asymptotic limit
# asymp_hack = np.zeros(len(detInts))
# for i in range(len(detInts)):
# 	asymp_hack[i] = ampInts[np.argmin(np.abs(0.1 - fidInts[:,i]))]

# figf = plt.figure()
# axf = figf.subplots()
# # axf.plot(detInts, asymp_lims)
# axf.set_xscale('log')
# # axf.set_yscale('log')
# axf.set_xlim(min(detInts), max(detInts))
# # axf.plot(detInts, asymp_fid_lims)
# # axf.plot(detInts, asymp_lims_03)
# # axf.plot(detInts, asymp_lims_1)
# axf.plot(detInts, asymp_hack)
# # axf.set_ylim(min(ampInts), max(ampInts))

#-----------------------------------------------------------------------------------------#
# Make plot
#-----------------------------------------------------------------------------------------#

# Define default colours list for manual cycling
# colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# mV to nT conversion factor:
ampToGauss = globs['Bz0_G_per_V']
mV_to_nT = ampToGauss*1e2

# Convert ampInts to nT for plotting:
ampInts = ampInts * mV_to_nT

fig1 = plt.figure(figsize=(9,9))
ax1 = fig1.subplots()
im1 = ax1.imshow(fidInts, vmin = 0.0001, vmax = 0.1, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1))
# im = ax.imshow(fidInts, vmin = 0.0001, vmax = 0.03, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1))
fig1.colorbar(im1, ax = ax1, shrink = 0.73, pad = 0.02, label = r' Transition probability $P_{LZ}$')

# Mark limits
ax1.axhline(adiabaticAmpLim*mV_to_nT, ls = '--', c = 'm', label = 'adiabatic limit')
ax1.plot(detInts, asymp_lims*mV_to_nT, ls = '--', c = 'crimson', label = 'asymptotic limit')
ax1.axhline(rabi_to_amp(150)*mV_to_nT, ls = '--', c = 'k', label = 'second RWA limit')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(min(detInts), max(detInts))
ax1.set_ylim(min(ampInts), max(ampInts))
ax1.set_ylabel('Signal amplitude [nT]')
ax1.set_xlabel('Initial detuning [Hz]')

# Annotate regions
# ax1.text(250, 3, 'non-adiabatic, \nnon-asymptotic', c = 'crimson')
# ax1.text(130, 0.7, 'adiabatic, \nnon-asymptotic', c = 'orangered')
# ax1.text(3800, 3, 'non-adiabatic, \nasymptotic', c = 'orangered')
# ax1.text(3800, 0.25, 'adiabatic, \nasymptotic', c = 'limegreen')
ax1.text(250, 3*mV_to_nT, 'non-adiabatic, \nnon-asymptotic', c = 'crimson')
ax1.text(130, 0.7*mV_to_nT, 'adiabatic, \nnon-asymptotic', c = 'orangered')
ax1.text(3800, 3*mV_to_nT, 'non-adiabatic, \nasymptotic', c = 'orangered')
ax1.text(3800, 0.25*mV_to_nT, 'adiabatic, \nasymptotic', c = 'limegreen')

ax1.legend()
ax1.set_title('Landau-Zener transition probability map')

# Save figures
print('Plotting complete, outputs saved to ' + str(folder))
plt.savefig(os.path.join(folder, 'fidelityMap.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'fidelityMap.pdf'), dpi = 300, bbox_inches='tight')

# fig4 = plt.figure(figsize=(9,9))
# ax4 = fig4.subplots()
# im4 = ax4.imshow(fidInts, vmin = 0.0001, vmax = 0.1, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1), zorder = 1)
# # im = ax.imshow(fidInts, vmin = 0.0001, vmax = 0.03, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1))
# fig4.colorbar(im4, ax = ax4, shrink = 0.73, pad = 0.02, label = r' Transition probability $P_{LZ}$')

# # Mark limits
# ax4.scatter(np.array(amps)*1e3, np.array(freqs) - fsweep_actual[0], marker = 'o', c = 'slategrey', label = 'dualspace data', zorder = 2)
# ax4.axhline(adiabaticAmpLim, ls = '--', c = 'm', label = 'adiabatic limit')
# ax4.plot(detInts, asymp_lims, ls = '--', c = 'crimson', label = 'asymptotic limit')
# ax4.axhline(rabi_to_amp(150), ls = '--', c = 'k', label = 'metrology limit')

# ax4.set_xscale('log')
# ax4.set_yscale('log')
# ax4.set_xlim(min(detInts), max(detInts))
# ax4.set_ylim(min(ampInts), max(ampInts))
# ax4.set_ylabel('Signal amplitude [mV]')
# ax4.set_xlabel('Initial detuning [Hz]')

# # Annotate regions
# ax4.text(250, 3, 'non-adiabatic, \nnon-asymptotic', c = 'crimson')
# ax4.text(130, 0.7, 'adiabatic, \nnon-asymptotic', c = 'orangered')
# ax4.text(3800, 3, 'non-adiabatic, \nasymptotic', c = 'orangered')
# ax4.text(3800, 0.25, 'adiabatic, \nasymptotic', c = 'limegreen')

# ax4.legend()
# ax4.set_title('Landau-Zener transition probablity map')

# # Save figures
# print('Plotting complete, outputs saved to ' + str(folder))
# plt.savefig(os.path.join(folder, 'fidelityMap_scatter.png'), dpi = 300, bbox_inches='tight')
# plt.savefig(os.path.join(folder, 'fidelityMap_scatter.pdf'), dpi = 300, bbox_inches='tight')

# fig2 = plt.figure(figsize=(9,9))
# ax2 = fig2.subplots()
# im2 = ax2.imshow(fidIntsAd, vmin = 0.0001, vmax = 0.1, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1))
# # im = ax.imshow(fidInts, vmin = 0.0001, vmax = 0.03, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1))
# fig2.colorbar(im2, ax = ax2, shrink = 0.84, pad = 0.02, label = r' Transition probability $P_{LZ}$')

# # Mark limits
# ax2.axhline(adiabaticAmpLim, ls = '--', c = 'm', label = 'adiabatic lim')
# ax2.plot(detInts, asymp_lims, ls = '--', c = 'crimson', label = 'asymptotic lim')
# ax2.axhline(rabi_to_amp(150), ls = '--', c = 'k', label = 'metrology lim')

# ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.set_xlim(min(detInts), max(detInts))
# ax2.set_ylim(min(ampInts), max(ampInts))
# ax2.set_ylabel('Signal amplitude [mV]')
# ax2.set_xlabel('Initial detuning [Hz]')
# ax2.legend()
# ax2.set_title('Landau-Zener transition probablity map - adiabatic')

# fig3 = plt.figure(figsize=(9,9))
# ax3 = fig3.subplots()
# im3 = ax3.imshow(fidIntsAs, vmin = 0.0001, vmax = 0.1, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)], norm = LogNorm(vmin = 0.0001, vmax = 0.1))
# # im3 = ax3.imshow(fidIntsAs, vmin = 0.0001, vmax = 0.1, extent = [min(detInts), max(detInts), min(ampInts), max(ampInts)])
# fig3.colorbar(im3, ax = ax3, shrink = 0.84, pad = 0.02, label = r' Transition probability $P_{LZ}$')

# # Mark limits
# ax3.axhline(adiabaticAmpLim, ls = '--', c = 'm', label = 'adiabatic lim')
# ax3.plot(detInts, asymp_lims, ls = '--', c = 'crimson', label = 'asymptotic lim')
# # ax3.plot(detInts, asymp_hack, ls = '--', c = 'crimson', label = 'asymptotic lim')
# ax3.axhline(rabi_to_amp(150), ls = '--', c = 'k', label = 'metrology lim')

# ax3.set_xscale('log')
# ax3.set_yscale('log')
# ax3.set_xlim(min(detInts), max(detInts))
# ax3.set_ylim(min(ampInts), max(ampInts))
# ax3.set_ylabel('Signal amplitude [mV]')
# ax3.set_xlabel('Initial detuning [Hz]')
# ax3.legend()
# ax3.set_title('Landau-Zener transition probablity map - asymptotic')

plt.show()

# # Save figures
# print('Plotting complete, outputs saved to ' + str(folder))
# plt.savefig(os.path.join(folder, 'fidelityMap.png'), dpi = 300, bbox_inches='tight')
# plt.savefig(os.path.join(folder, 'fidelityMap.pdf'), dpi = 300, bbox_inches='tight')