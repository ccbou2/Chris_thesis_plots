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
subdf = df.loc['20191008T171321']
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
subdf.set_index([df.tuplify('magnetom_rabi_sweep_initial')], inplace = True)

#-----------------------------------------------------------------------------------------#
# Import and process lockin traces
#-----------------------------------------------------------------------------------------#

subRange = 0.25
datadf = pd.DataFrame(columns = ['det_init', 't_lockin', 'Q_lockin', 'res_time', 'res_freq'])
detList = []
rateList = []

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
	ac_amp = globs['magnetom_rabi_spectrum_amps'][0]
	ac_freq = globs['magnetom_rabi_spectrum_freqs'][0]
	fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

	# Compute actual bounds of sweep
	amp_bounds = (better_amp_calib(fsweep_requested[0]), better_amp_calib(fsweep_requested[1]))
	fsweep_actual = (poly_freq_calib(amp_bounds[0]), poly_freq_calib(amp_bounds[1]))
	rabi_range = abs(fsweep_actual[1] - fsweep_actual[0])
	det_init = ac_freq - fsweep_actual[0]
	sweepRate = rabi_range/T
	detList.append(det_init)
	rateList.append(sweepRate)

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

	# # Readjust times such that t_res = 0, for plotting overlay
	# t_adjust_sub = t_lockin_sub - ac_res_time_lockin

	# Write filter_Q to dataframe, with freq as the row index
	dataSeries = pd.DataFrame({'det_init': det_init, 't_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq})
	# ampDatadf.append({'t_lockin':t_lockin_sub, 'Q_lockin':filter_Q_sub, 'res_time': ac_res_time_lockin, 'res_freq':ac_res_rabi_freq}, name = freq)
	datadf = datadf.append(dataSeries)

datadf.set_index('det_init', inplace = True)

#-----------------------------------------------------------------------------------------#
# Make plot
#-----------------------------------------------------------------------------------------#

# Define default colours list for manual cycling
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Introduce figure object and title
fig = plt.figure('detuningSweeps', figsize = (9, 14))

xlim = [ac_res_time_lockin-0.045, ac_res_time_lockin+0.035]

# Counter for iteration
i = 1
for det, rate in zip(detList, rateList):
	# make subplot
	ax = fig.add_subplot(len(detList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	# ax.plot(ampDatadf.loc[(det,'t_lockin')], ampDatadf.loc[(det,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta_i$ = ' + str(det/1e3) + ' kHz, ' + r'$\lambda$ = ' + str(rate) + ' kHz/s')
	ax.plot(datadf.loc[(det,'t_lockin')], datadf.loc[(det,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta_i$ = {:.2f} kHz, $\lambda$ = {:.2f} kHz/s'.format(det/1e3, rate/1e3))
	# Set subplot bounds
	ax.set_xlim(xlim)
	ax.set_ylim([-0.85, 0.85])
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(detList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 4:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		plt.title('Transition dependence on sweep bandwidth')
	plt.legend(loc = 'upper right')
	# plt.legend(loc = 'lower left')
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Time [s]')
# Save figures
plt.savefig(os.path.join(folder, 'bandwidthQcompare.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'bandwidthQcompare.pdf'), dpi = 300, bbox_inches='tight')

# Introduce figure object and title
fig = plt.figure('detuningSweeps2', figsize = (9, 14))

# Counter for iteration
i = 1
for det, rate in zip(detList, rateList):
	# make subplot
	ax = fig.add_subplot(len(detList), 1, i)
	# Add mostly transparent abscissa at y = 0
	ax.axhline(0, c = 'lightgray', alpha = 0.5)
	# Plot data into subplot with cyclic colour and label
	# ax.plot(ampDatadf.loc[(det,'t_lockin')], ampDatadf.loc[(det,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta_i$ = ' + str(det/1e3) + ' kHz, ' + r'$\lambda$ = ' + str(rate) + ' kHz/s')
	ax.plot(datadf.loc[(det,'t_lockin')], datadf.loc[(det,'Q_lockin')]*1e3, c = colours[i-1], label = r'$\Delta_i$ = {:.2f} kHz, $\lambda$ = {:.2f} kHz/s'.format(det/1e3, rate/1e3))
	# Set subplot bounds
	ax.set_xlim([ac_res_time_lockin-0.045, ac_res_time_lockin+0.035])
	ax.set_ylim([-0.85, 0.85])
	# Boolean operations to remove frames and axes on correct subplots
	if i != len(detList):
		ax.xaxis.set_visible(False)
		ax.spines["bottom"].set_visible(False)
	if i != 1:
		ax.spines["top"].set_visible(False)
	if i == 4:
		ax.set_ylabel('Q(t) [mV]')
	if i == 1:
		plt.title('Transition dependence on sweep bandwidth')
	# plt.legend(loc = 'upper right')
	plt.legend(loc = 'lower left')
	i += 1
# Remove space between subplots and add xlabel
plt.subplots_adjust(hspace=0)
plt.xlabel('Time [s]')
# Save figures
plt.savefig(os.path.join(folder, 'bandwidthQcompare2.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'bandwidthQcompare2.pdf'), dpi = 300, bbox_inches='tight')

plt.show()
print('Plotting complete, outputs saved to ' + str(folder))