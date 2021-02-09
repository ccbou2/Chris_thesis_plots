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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
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
subdf = df.loc['20190925T150456']
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
# subdf.set_index([df.tuplify('magnetom_ac_amplitude'), df.tuplify('magnetom_ac_frequency')], inplace = True)

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]

#-----------------------------------------------------------------------------------------#
# Process data
#-----------------------------------------------------------------------------------------#

path = subdf['filepath'][0]
dataset = data(path)
run = Run(path)
shot_id = os.path.split(path)[-1].replace('.h5', '')

# Load spectrogram and lockin Q channel from hdf5
t, f, stft, globs = load_stft(path, all_globals=True)
t_lockin, lockin_Q, lockin_globs = get_lockin_trace(path, '/data/traces/lockin_connection/Y_channel')

if stft is None:
	pass
	print(Exception('No STFT present'))
else:
	# Analysis parameters
	f_sp = globs['sp_center_freq']*1e6

#-----------------------------------------------------------------------------------------#
# Rabi frequency sweep analysis
#-----------------------------------------------------------------------------------------#

# Set subrange for limiting later
subRange = 0.3

# import lockin trace
t_lockin, lockin_Q, lockin_globs = get_lockin_trace(path)

# retrieve rest of globals
globs = import_globs(path)

# Define relevant params from lockin_globs
lockin_sample_rate = lockin_globs['actual capture sample rate (Hz)']
filter_cutoff = 2e3
t0 = globs['hobbs_settle'] + globs['magnetom_ac_wait']
signalT = globs['magnetom_ac_duration']
# rabiT = globs['magnetom_rabi_sweep_duration']
tsweep = (t0, t0 + signalT)
# fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

# Signal params
# ac_amp = globs['magnetom_ac_amplitude']
# ac_freq = globs['magnetom_ac_frequency']
ac_amp = globs['magnetom_ac_amplitude']
ac_freq = globs['magnetom_ac_frequency']
# fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

# Compute actual bounds of sweep
# amp_bounds = (bad_amp_calib(fsweep_requested[0]), bad_amp_calib(fsweep_requested[1]))
# amp_bounds = (amp_calib(fsweep_requested[0], shot_id), amp_calib(fsweep_requested[1], shot_id))
# fsweep_actual = (poly_freq_calib(amp_bounds[0]), poly_freq_calib(amp_bounds[1]))
# rabi_range = abs(fsweep_actual[1] - fsweep_actual[0])

# apply butterworth filter (same as rabi_sweep)
filter_b, filter_a = signal.butter(8, 2 * filter_cutoff / lockin_sample_rate)
filter_Q = signal.filtfilt(filter_b, filter_a, lockin_Q)

# # Repeat Rabi sweep frequency calibrations for lockin trace:
# amp_map = np.vectorize(amp_map)
# rf_amps_lockin = amp_map(t_lockin, t0, rabiT, amp_bounds[0], amp_bounds[1])
# rabi_freqs_lockin = poly_freq_calib(rf_amps_lockin)
# ac_res_time_lockin = t_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]
# ac_res_rabi_freq = rabi_freqs_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]

# # Restrict data to desired range for plotting
# filter_Q_sub = filter_Q[np.logical_and(rabi_freqs_lockin > ac_res_rabi_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_res_rabi_freq + rabi_range*subRange/2)]
# t_lockin_sub = t_lockin[np.logical_and(rabi_freqs_lockin > ac_res_rabi_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_res_rabi_freq + rabi_range*subRange/2)]
# rabi_freqs_sub = poly_freq_calib(amp_map(t_lockin_sub, t0, rabiT, amp_bounds[0], amp_bounds[1]))

# Readjust times such that t_res = 0, for plotting overlay
# t_adjust_sub = t_lockin_sub - ac_res_time_lockin

# Compute sweep bounds (time and freq) and predicted resonances from AC signals for lockin
bound_times_lockin = [t_lockin[np.argmin(abs(t_lockin - t_b))] for t_b in tsweep]
# bound_freqs_lockin = [rabi_freqs_lockin[np.argmin(abs(t_lockin - t_b))] for t_b in tsweep]
# # set up secondary axis ticks
# rabi_tick_locs = [bound_times_lockin[0]]
# rabi_tick_freqs = ["%0.2f" % (bound_freqs_lockin[0]/1e3)]
# rabi_tick_locs.append(ac_res_time_lockin)
# rabi_tick_freqs.append("%0.2f" % (ac_res_rabi_freq/1e3))
# rabi_tick_locs.append(bound_times_lockin[1])
# rabi_tick_freqs.append("%0.2f" % (bound_freqs_lockin[1]/1e3))

# Compute periodogram
Q_f, Q_Pxx_spec = signal.periodogram(lockin_Q[np.logical_and(t_lockin > tsweep[0], t_lockin < tsweep[1])], lockin_sample_rate, 'flattop', scaling='spectrum')

#-----------------------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Define some plot limits
larmor = globs['sp_center_freq']*1e6
print(larmor)
# spec_ylims = np.array((larmor - np.max(rabi_freqs_sub), larmor + np.max(rabi_freqs_sub)))/1e3
spec_ylims = np.array((larmor - 12e3, larmor + 12e3))/1e3
# demod_ylims = (-max(filter_Q_sub)*1e3-0.05, max(filter_Q_sub)*1e3+0.05)
demod_ylims = (-max(filter_Q)*1e3-0.05, max(filter_Q)*1e3+0.05)
# time_lims = (np.min(t_lockin_sub), np.max(t_lockin_sub))
# time_lims_1 = (np.min(t_lockin), np.max(t_lockin))
time_lims_1 = (np.min(t_lockin), 0.39)
time_lims_zoom = (0.10, 0.125)
trange1 = time_lims_1[1] - time_lims_1[0]
trange_zoom = time_lims_zoom[1] - time_lims_zoom[0]
Qrange = demod_ylims[1] - demod_ylims[0]
spectrogram_lims = [1, 3.5]

# Define figure object
fig = plt.figure('dressed_magnetom_1', figsize = (10,6))
gsTitle = fig.add_gridspec(1,1)
gstight = fig.add_gridspec(3,2, hspace = 0)
gsother = fig.add_gridspec(3,2, hspace = 0.4)

axTitle = fig.add_subplot(gsTitle[:,:])
axTitle.set_title('Dressed AC magnetometry - 2mV signal', y = 1.04)
axTitle.axis('off')

# Plot and label spectrogram
ax1 = fig.add_subplot(gstight[0,0])
# ax1.plot_stft(stft, t, f, range=spectrogram_lims, png=None)
# ax1.set_title('Concatenated dynamical decoupling')
ax1.imshow(np.log10(stft.T), origin='lower', vmin=spectrogram_lims[0], vmax=spectrogram_lims[1], extent=[t[0], t[-1], f[0]/1e3, f[-1]/1e3], aspect='auto', interpolation='nearest', cmap='jet')
# ax1.axvline(ac_res_time_lockin, c = 'm', ls = '--')
ax1.axvline(bound_times_lockin[0], c = 'orange', ls = '--')
ax1.axvline(bound_times_lockin[1], c = 'orange', ls = '--')
# ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [kHz]')
ax1.set_xlim(time_lims_1)
ax1.set_ylim(spec_ylims)
ax1.xaxis.set_visible(False)
ax1.spines["bottom"].set_visible(False)

# Make second subplot for demod channel, w/ fits
ax2 = fig.add_subplot(gstight[1,0])
ax2.spines["top"].set_visible(False)
ax2.plot(t_lockin, filter_Q*1e3, label = 'filtered Q(t)')
ax2.axhline(0, c = 'lightgray', alpha = 0.5)
# ax2.plot(t_lockin_sub, adiabaticFitRecon*1e3, ls = '--', label = 'adiabatic fit')
# ax2.plot(t_lockin_sub, analyticFitRecon*1e3, ls = '--', c = 'k', label = 'analytic fit')
# ax2.axvline(ac_res_time_lockin, c = 'm', ls = '--', label = r'$f_s$ = {:.2f} kHz'.format(ac_res_rabi_freq/1e3))
ax2.axvline(bound_times_lockin[0], c = 'orange', ls = '--', label = 'Signal bounds', alpha = 0.6)
ax2.axvline(bound_times_lockin[1], c = 'orange', ls = '--', alpha = 0.6)
# ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Q(t) [mV]')
ax2.set_ylim(demod_ylims)
ax2.set_xlim(time_lims_1)
# ax2.legend(loc = 'lower center')
ax2.legend(loc = 'upper right')
# plt.subplots_adjust(hspace = 0)

ax3 = fig.add_subplot(gsother[2,0])
ax3.axhline(0, c = 'lightgray', alpha = 0.5)
ax3.plot(t_lockin, filter_Q*1e3, label = 'filtered Q(t)')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Q(t) [mV]')
ax3.set_ylim(demod_ylims)
ax3.set_xlim(time_lims_zoom)
mark_inset(ax2, ax3, loc1=1, loc2=2, fc="crimson", alpha = 0.3, ec='crimson', ls='dashdot', lw = 1)

ax4 = fig.add_subplot(gsother[:,1])
ax4.plot(Q_f, Q_Pxx_spec)
ax4.set_xlim(0,500)
ax4.set_xlabel('Frequency [Hz]')
ax4.set_ylabel('Periodogram amplitude')

# # axins = inset_axes(ax2, width=2.2, height = 0.8, loc = 4, borderpad = 1.7)
# axins = inset_axes(ax2, width = 2.6, height = 0.8, loc = 8, borderpad = 1.9)
# axins.axis([0.1, 0.105, -0.3, 0.4])
# axins.set_yticklabels(['','','0'])
# axins.axhline(0, c = 'lightgray', alpha = 0.5)
# axins.plot(t_lockin[np.logical_and(t_lockin > 0.1 , t_lockin < 0.105)], filter_Q[np.logical_and(t_lockin > 0.1 , t_lockin < 0.105)]*1e3, c = colours[0])
# mark_inset(ax2, axins, loc1=1, loc2=3, fc="crimson", alpha = 0.2, ec='crimson', ls='dashdot', lw = 1)

# # Define ticks for Rabi frequency
# rabi_tick_locs = ax2.get_xticks()
# rabi_tick_freqs = np.around(poly_freq_calib(amp_map(rabi_tick_locs, t0, rabiT, amp_bounds[0], amp_bounds[1]))/1e3, 2)

# # Make secondary axis for Rabi frequency on spectrogram
# ax22 = ax1.twiny()
# ax22.set_xticks(rabi_tick_locs)
# ax22.set_xticklabels(rabi_tick_freqs)
# ax22.set_xlim(ax2.get_xlim())
# ax22.set_xlabel(r'$\Omega_1$ [kHz]')
# ax1.xaxis.set_visible(False)
# ax1.spines["bottom"].set_visible(False)

# # Convert text coords into positions based on plot ranges
# def textCoords_x(coord, tmin=time_lims[0], trange=trange):
# 	return tmin + coord*trange
# def textCoords_y(coord, Qmin=demod_ylims[0], Qrange=Qrange):
# 	return Qmin + coord*Qrange

# Add results text to plot
# ax2.text(0.21, 0.25, 'fit results:', c = colours[0])
# ax2.text(0.21, -0.1, result_string_ad, c = colours[1])
# ax2.text(0.21, -0.4, result_string_an, c = 'k')
# ax2.text(0.21, -0.7, result_string_exp, c = 'g')
# ax2.text(textCoords_x(0.042), textCoords_y(0.587), 'fit results:', c = colours[0])
# ax2.text(textCoords_x(0.042), textCoords_y(0.411), result_string_ad, c = colours[1])
# ax2.text(textCoords_x(0.042), textCoords_y(0.245), result_string_an, c = 'k')
# ax2.text(textCoords_x(0.042), textCoords_y(0.088), result_string_exp, c = 'g')
# textObjs = [t1,t2,t3]

# txmin = min([txt.get_window_extent().xmin for txt in textObjs])
# txmax = min([txt.get_window_extent().xmax for txt in textObjs])
# tymin = min([txt.get_window_extent().ymin for txt in textObjs])
# tymax = min([txt.get_window_extent().ymax for txt in textObjs])

# txmin, tymin = fig.transFigure.inverted().transform((txmin, tymin))
# txmax, tymax = fig.transFigure.inverted().transform((txmax, tymax))
# txbuffer = 0.15*(txmax - txmin)

# ax2.add_patch(patches.Rectangle((txmin-txbuffer/2, tymin-txbuffer/2), txmax-txmin+txbuffer, tymax-tymin+txbuffer, facecolor = 'lightgray', edgecolor = 'k', alpha = 0.5, transform = fig.transFigure))
# ax2.add_patch(patches.Rectangle((0.208, -0.50), 0.023, 0.9, facecolor = 'white', edgecolor = 'k', alpha = 0.5))
# ax2.add_patch(patches.Rectangle((0.208, -0.75), 0.023, 1.15, facecolor = 'white', edgecolor = 'k', alpha = 0.5))
# ax2.add_patch(patches.Rectangle((textCoords_x(0.03), textCoords_y(0.0588)), 0.192*trange, 0.62*Qrange, facecolor = 'lightgrey', edgecolor = 'k', alpha = 0.5))

plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'dressed_magnetom_1.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'dressed_magnetom_1.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))