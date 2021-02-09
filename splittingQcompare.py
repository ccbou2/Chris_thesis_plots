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
# subdf = df.loc[['20191008T133751','20191008T133800','20191008T133805','20191008T133830','20191008T145930', '20191008T133820']]
subdf = df.loc[['20191008T133438','20191008T133707','20191008T133713','20191008T133717','20191008T133722','20191008T133731']]
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
subdf.set_index([df.tuplify('magnetom_rabi_sweep_initial')], inplace = True)

#-----------------------------------------------------------------------------------------#
# Import and process lockin traces
#-----------------------------------------------------------------------------------------#

subRange = 0.55
datadf = pd.DataFrame(columns = ['f_split', 't_lockin', 'Q_lockin', 'res_time', 'res_freq'])
splitList = []

# Create a list of the paths for each shot in ampdf, and return the filtered Q channel for each
for f_init, row in subdf.iterrows():
	# print(freq, row['filepath'])
	path = row['filepath'][0]

	# import lockin trace for each
	t_lockin, lockin_Q, lockin_globs = get_lockin_trace(path)

	# retrieve rest of globals
	globs = import_globs(path)

	# Define relevant params from lockin_globs
	lockin_sample_rate = lockin_globs['actual capture sample rate (Hz)']
	filter_cutoff = 1e3
	t0 = globs['hobbs_settle'] + globs['magnetom_ac_wait']
	T = globs['magnetom_rabi_sweep_duration']
	tsweep = (t0, t0 + T)

	# Signal params
	ac_freq = globs['magnetom_rabi_spectrum_freqs'][0]
	ac_freq_split = globs['magnetom_rabi_spectrum_freqs'][1] - ac_freq
	fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

	# Compute actual bounds of sweep
	amp_bounds = (bad_amp_calib(fsweep_requested[0]), bad_amp_calib(fsweep_requested[1]))
	fsweep_actual = (poly_freq_calib(amp_bounds[0]), poly_freq_calib(amp_bounds[1]))
	rabi_range = abs(fsweep_actual[1] - fsweep_actual[0])
	splitList.append(ac_freq_split)

	# apply butterworth filter (same as rabi_sweep)
	filter_b, filter_a = signal.butter(8, 2 * filter_cutoff / lockin_sample_rate)
	filter_Q = signal.filtfilt(filter_b, filter_a, lockin_Q)

	# Repeat Rabi sweep frequency calibrations for lockin trace:
	amp_map = np.vectorize(amp_map)
	rf_amps_lockin = amp_map(t_lockin, t0, T, amp_bounds[0], amp_bounds[1])
	rabi_freqs_lockin = poly_freq_calib(rf_amps_lockin)
	ac_res_time_lockin = t_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]
	ac_res_rabi_freq = rabi_freqs_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]

	# Restrict data to desired range for plotting
	filter_Q_sub = filter_Q[np.logical_and(rabi_freqs_lockin > ac_freq + 1500 - rabi_range*subRange/2, rabi_freqs_lockin < ac_freq + 1500 + rabi_range*subRange/2)]
	t_lockin_sub = t_lockin[np.logical_and(rabi_freqs_lockin > ac_freq + 1500 - rabi_range*subRange/2, rabi_freqs_lockin < ac_freq + 1500 + rabi_range*subRange/2)]

	# # Readjust times such that t_res = 0, for plotting overlay
	# t_adjust_sub = t_lockin_sub - ac_res_time_lockin

	# Write filter_Q to dataframe, with freq as the row index
	dataSeries = pd.DataFrame({'f_split': ac_freq_split, 't_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq})
	datadf = datadf.append(dataSeries)

datadf.set_index('f_split', inplace = True)
splitList.sort()

#-----------------------------------------------------------------------------------------#
# Make plot
#-----------------------------------------------------------------------------------------#

# Define default colours list for manual cycling
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

tCentre = ac_res_time_lockin + 0.5 * 0.4 * max(splitList) / rabi_range
xlims = [tCentre-0.1, tCentre+0.1]

# # Introduce figure object and title
# fig = plt.figure('freqSplit', figsize = (9, 14))

# # Counter for iteration
# i = 1
# for delf in splitList:
# 	# make subplot
# 	ax = fig.add_subplot(len(splitList), 1, i)
# 	# Add mostly transparent abscissa at y = 0
# 	ax.axhline(0, c = 'lightgray', alpha = 0.5)
# 	# Plot data into subplot with cyclic colour and label
# 	ax.plot(datadf.loc[(delf,'t_lockin')], datadf.loc[(delf,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta f$ = {:.2f} kHz'.format(delf/1e3))
# 	# Set subplot bounds
# 	ax.set_xlim(xlims)
# 	ax.set_ylim([-0.85, 0.85])
# 	# Boolean operations to remove frames and axes on correct subplots
# 	if i != len(splitList):
# 		ax.xaxis.set_visible(False)
# 		ax.spines["bottom"].set_visible(False)
# 	if i != 1:
# 		ax.spines["top"].set_visible(False)
# 	if i == 4:
# 		ax.set_ylabel('Q(t) [mV]')
# 	if i == 1:
# 		plt.title('Dependence on transition splitting for dual-tone signals')
# 	plt.legend(loc = 'upper right')
# 	# plt.legend(loc = 'lower left')
# 	i += 1
# # Remove space between subplots and add xlabel
# plt.subplots_adjust(hspace=0)
# plt.xlabel('Time [s]')
# # Save figures
# plt.savefig(os.path.join(folder, 'splittingQcompare.png'), dpi = 300, bbox_inches='tight')
# plt.savefig(os.path.join(folder, 'splittingQcompare.pdf'), dpi = 300, bbox_inches='tight')

# Introduce figure object and title
fig = plt.figure('freqSplit_secax', figsize = (9, 14))

# Counter for iteration
i = 1
for delf in splitList:
	# make subplot
	ax = fig.add_subplot(len(splitList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	ax.plot(datadf.loc[(delf,'t_lockin')], datadf.loc[(delf,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta f$ = {:.2f} kHz'.format(delf/1e3))
	# Set subplot bounds
	ax.set_xlim(xlims)
	ax.set_ylim([-0.85, 0.85])
	plt.legend(loc = 'upper right')
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(splitList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 4:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		ax.set_title('Dependence on frequency separation for dual-tone signals')
		# Define ticks for Rabi frequency
		rabi_tick_locs = ax.get_xticks()
		rabi_tick_freqs = np.around(poly_freq_calib(amp_map(rabi_tick_locs, t0, T, amp_bounds[0], amp_bounds[1]))/1e3, 2)
		# Make secondary axis for Rabi frequency on spectrogram
		ax2 = ax.twiny()
		ax2.set_xticks(rabi_tick_locs)
		ax2.set_xticklabels(rabi_tick_freqs)
		ax2.set_xlim(xlims)
		ax2.set_xlabel(r'$\Omega_{{1R}}$ [kHz]')
		ax.spines["bottom"].set_visible(False)
		ax.spines["top"].set_visible(False)
		ax2.spines["bottom"].set_visible(False)
		ax2.spines["left"].set_visible(False)
		ax2.spines["right"].set_visible(False)
	# plt.legend(loc = 'lower left')
	if i == 2:
		ax.spines["top"].set_visible(False)
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Time [s]')
# Save figures
plt.savefig(os.path.join(folder, 'splittingQcompare_secax.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'splittingQcompare_secax.pdf'), dpi = 300, bbox_inches='tight')

# # Introduce figure object and title
# fig = plt.figure('freqSplit2', figsize = (9, 14))

# # Counter for iteration
# i = 1
# for delf in splitList:
# 	# make subplot
# 	ax = fig.add_subplot(len(splitList), 1, i)
# 	# Add mostly transparent abscissa at y = 0
# 	ax.axhline(0, c = 'lightgray', alpha = 0.5)
# 	# Plot data into subplot with cyclic colour and label
# 	ax.plot(datadf.loc[(delf,'t_lockin')], datadf.loc[(delf,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta f$ = {:.2f} kHz'.format(delf/1e3))
# 	# Set subplot bounds
# 	ax.set_xlim(xlims)
# 	ax.set_ylim([-0.85, 0.85])
# 	# Boolean operations to remove frames and axes on correct subplots
# 	if i != len(splitList):
# 		ax.xaxis.set_visible(False)
# 		ax.spines["bottom"].set_visible(False)
# 	if i != 1:
# 		ax.spines["top"].set_visible(False)
# 	if i == 4:
# 		ax.set_ylabel('Q(t) [mV]')
# 	if i == 1:
# 		plt.title('Dependence on transition splitting for dual-tone signals')
# 	# plt.legend(loc = 'upper right')
# 	plt.legend(loc = 'lower left')
# 	i += 1
# # Remove space between subplots and add xlabel
# plt.subplots_adjust(hspace=0)
# plt.xlabel('Time [s]')
# # Save figures
# plt.savefig(os.path.join(folder, 'splittingQcompare2.png'), dpi = 300, bbox_inches='tight')
# plt.savefig(os.path.join(folder, 'splittingQcompare2.pdf'), dpi = 300, bbox_inches='tight')

# Introduce figure object and title
fig = plt.figure('freqSplit2_secax', figsize = (9, 14))

# Counter for iteration
i = 1
for delf in splitList:
	# make subplot
	ax = fig.add_subplot(len(splitList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	ax.plot(datadf.loc[(delf,'t_lockin')], datadf.loc[(delf,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta f$ = {:.2f} kHz'.format(delf/1e3))
	# Set subplot bounds
	ax.set_xlim(xlims)
	ax.set_ylim([-0.85, 0.85])
	plt.legend(loc = 'lower left')
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(splitList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 4:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		plt.title('Dependence on frequency separation for dual-tone signals')
		# Define ticks for Rabi frequency
		rabi_tick_locs = ax.get_xticks()
		rabi_tick_freqs = np.around(poly_freq_calib(amp_map(rabi_tick_locs, t0, T, amp_bounds[0], amp_bounds[1]))/1e3, 2)
		# Make secondary axis for Rabi frequency on spectrogram
		ax2 = ax.twiny()
		ax2.set_xticks(rabi_tick_locs)
		ax2.set_xticklabels(rabi_tick_freqs)
		ax2.set_xlim(xlims)
		ax2.set_xlabel(r'$\Omega_1$ [kHz]')
		ax.spines["bottom"].set_visible(False)
		ax.spines["top"].set_visible(False)
		ax2.spines["bottom"].set_visible(False)
		ax2.spines["left"].set_visible(False)
		ax2.spines["right"].set_visible(False)
	if i == 2:
		ax.spines["top"].set_visible(False)
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Time [s]')
# Save figures
plt.savefig(os.path.join(folder, 'splittingQcompare2_secax.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'splittingQcompare2_secax.pdf'), dpi = 300, bbox_inches='tight')

plt.show()
print('Plotting complete, outputs saved to ' + str(folder))