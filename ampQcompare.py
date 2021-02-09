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
subdf = df.loc['20191007T164123']
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
subdf.set_index([df.tuplify('magnetom_ac_amplitude'), df.tuplify('magnetom_ac_frequency')], inplace = True)

# Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
freqdf = subdf.loc[(slice(None), 9000.0), :]

#-----------------------------------------------------------------------------------------#
# Import and process lockin traces
#-----------------------------------------------------------------------------------------#

subRange = 0.35
freqDatadf = pd.DataFrame(columns = ['amp', 't_lockin', 'Q_lockin', 'res_time', 'res_freq'])
ampList = []

# Create a list of the paths for each shot in ampdf, and return the filtered Q channel for each
for amp, row in freqdf.iterrows():
	# print(freq, row['filepath'])
	path = row['filepath'][0]
	# print(amp)
	ampList.append(amp[0])

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

	# Write filter_Q to dataframe, with freq as the row index
	dataSeries = pd.DataFrame({'amp': amp[0], 't_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq})
	# ampDatadf.append({'t_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq}, name = freq)
	freqDatadf = freqDatadf.append(dataSeries)

freqDatadf.set_index('amp', inplace = True)

#-----------------------------------------------------------------------------------------#
# Make plot
#-----------------------------------------------------------------------------------------#

# Define default colours list for manual cycling
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

xlims = [-0.06+ac_res_time_lockin, 0.06+ac_res_time_lockin]
# Introduce figure object and title
fig = plt.figure('amps', figsize = (9,14))

# Counter for iteration
i = 1
for amp in ampList:
	# make subplot
	ax = fig.add_subplot(len(ampList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	ax.plot(freqDatadf.loc[(amp,'t_lockin')], freqDatadf.loc[(amp,'Q_lockin')]*1e3, c = colours[i-1], label = str(amp*1e3) + ' mV')
	# Set subplot bounds
	ax.set_xlim(xlims)
	ax.set_ylim([-0.75, 0.75])
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(ampList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 4:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		plt.title('Transition dependence on signal amplitude')
	plt.legend(loc = 'upper right')
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Time [s]')
# Save figures
plt.savefig(os.path.join(folder, 'ampQcompare.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'ampQcompare.pdf'), dpi = 300, bbox_inches='tight')

# Introduce figure object and title
fig = plt.figure('amps_secax', figsize = (9,14))

# Counter for iteration
i = 1
for amp in ampList:
	# make subplot
	ax = fig.add_subplot(len(ampList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	ax.plot(freqDatadf.loc[(amp,'t_lockin')], freqDatadf.loc[(amp,'Q_lockin')]*1e3, c = colours[i-1], label = str(amp*1e3) + ' mV')
	plt.legend(loc = 'upper right')
	# Set subplot bounds
	ax.set_xlim(xlims)
	ax.set_ylim([-0.75, 0.75])
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(ampList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 4:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		ax.set_title('Transition dependence on signal amplitude')
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
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Time [s]')
# Save figures
plt.savefig(os.path.join(folder, 'ampQcompare_secax.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'ampQcompare_secax.pdf'), dpi = 300, bbox_inches='tight')

plt.show()
print('Plotting complete, outputs saved to ' + str(folder))