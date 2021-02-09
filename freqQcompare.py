#!python3
from __future__ import division
from lyse import *
# from pylab import *
import matplotlib.pyplot as plt
# import matplotlib as mpl
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
# rcParams['figure.figsize'] = (8.0, 14.0)

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

# Retrieve data based on a set ac amplitude (first index), set amp to desired selection (0.001mV for now)
ampdf = subdf.loc[0.001]

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]

#-----------------------------------------------------------------------------------------#
# Import and process lockin traces
#-----------------------------------------------------------------------------------------#

subRange = 0.25
ampDatadf = pd.DataFrame(columns = ['freq', 't_lockin', 'Q_lockin', 'res_time', 'res_freq'])
freqList = []

# Create a list of the paths for each shot in ampdf, and return the filtered Q channel for each
for freq, row in ampdf.iterrows():
	# print(freq, row['filepath'])
	path = row['filepath'][0]
	freqList.append(freq)

	# import lockin trace for each
	# print(row['filepath'][0])
	t_lockin, lockin_Q, lockin_globs = get_lockin_trace(path)

	# retrieve rest of globals
	globs = import_globs(path)

	# Define relevant params from lockin_globs
	lockin_sample_rate = lockin_globs['actual capture sample rate (Hz)']
	filter_cutoff = 1e3
	t0 = globs['hobbs_settle'] + globs['magnetom_ac_wait']
	T = globs['magnetom_rabi_sweep_duration']
	tsweep = (t0, t0 + T)
	fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

	# Signal params
	ac_amp = globs['magnetom_ac_amplitude']
	ac_freq = globs['magnetom_ac_frequency']
	fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

	# Compute actual bounds of sweep
	amp_bounds = (bad_amp_calib(fsweep_requested[0]), bad_amp_calib(fsweep_requested[1]))
	fsweep_actual = (poly_freq_calib(amp_bounds[0]), poly_freq_calib(amp_bounds[1]))
	rabi_range = abs(fsweep_actual[1] - fsweep_actual[0])

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
	filter_Q_sub = filter_Q[np.logical_and(rabi_freqs_lockin > ac_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_freq + rabi_range*subRange/2)]
	t_lockin_sub = t_lockin[np.logical_and(rabi_freqs_lockin > ac_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_freq + rabi_range*subRange/2)]

	# Readjust times such that t_res = 0, for plotting overlay
	t_adjust_sub = t_lockin_sub - ac_res_time_lockin

	# Write filter_Q to dataframe, with freq as the row index
	dataSeries = pd.DataFrame({'freq': freq, 't_lockin':t_adjust_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq})
	# ampDatadf.append({'t_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq}, name = freq)
	ampDatadf = ampDatadf.append(dataSeries)

ampDatadf.set_index('freq', inplace = True)

#-----------------------------------------------------------------------------------------#
# Make plot
#-----------------------------------------------------------------------------------------#

# Define default colours list for manual cycling
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Introduce figure object and title
fig = plt.figure(figsize = (9,14))
# fig = plt.figure()

xlim = [-0.04, 0.04]

# Counter for iteration
i = 1
for freq in freqList:
	# make subplot
	ax = fig.add_subplot(len(freqList), 1, i)
	# ax = plt.subplot(len(freqList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	ax.plot(ampDatadf.loc[(freq,'t_lockin')], ampDatadf.loc[(freq,'Q_lockin')]*1e3, c = colours[i-1], label = str(freq/1e3) + ' kHz')
	# Set subplot bounds
	ax.set_xlim(xlim)
	ax.set_ylim([-0.75, 0.75])
	plt.legend(loc = 'upper right')
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(freqList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 3:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		ax.set_title('Transition dependence on signal frequency')
		# Define ticks for Rabi frequency
		# rabi_tick_locs = ax.get_xticks()
		# rabi_tick_freqs = np.around(poly_freq_calib(amp_map(rabi_tick_locs + ac_res_time_lockin, t0, T, amp_bounds[0], amp_bounds[1]))/1e3, 2)
		# # Make secondary axis for Rabi frequency on spectrogram
		# ax2 = ax.twiny()
		# ax2.set_xticks(rabi_tick_locs)
		# ax2.set_xticklabels(rabi_tick_freqs)
		# ax2.set_xlim(xlim)
		# ax2.set_xlabel(r'$\Omega_1$ [kHz]')
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Rescaled time [s]')
plt.show()
# Save figures
plt.savefig(os.path.join(folder, 'freqQcompare.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'freqQcompare.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))