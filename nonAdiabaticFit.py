#!python3
from __future__ import division
from lyse import *
from pylab import *
import numpy as np
from analysislib.common import *
from analysislib.spinor.aliases import *
from analysislib.spinor.faraday.faraday_aux import load_stft, plot_stft
from analysislib.spinor.Chris_thesis_plots.func_lib import *
from analysislib.spinor.parcyl import parcyd, rhoVec, rhoVec_ang
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
subdf = df.loc['20191008T171321']
seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
# subdf.set_index([df.tuplify('magnetom_ac_amplitude'), df.tuplify('magnetom_ac_frequency')], inplace = True)

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]

#-----------------------------------------------------------------------------------------#
# Process data
#-----------------------------------------------------------------------------------------#

path = subdf['filepath'][2]
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
filter_cutoff = 1e3
t0 = globs['hobbs_settle'] + globs['magnetom_ac_wait']
rabiT = globs['magnetom_rabi_sweep_duration']
tsweep = (t0, t0 + rabiT)
fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

# Signal params
# ac_amp = globs['magnetom_ac_amplitude']
# ac_freq = globs['magnetom_ac_frequency']
ac_amp = globs['magnetom_rabi_spectrum_amps'][0]
ac_freq = globs['magnetom_rabi_spectrum_freqs'][0]
fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

# Compute actual bounds of sweep
# amp_bounds = (bad_amp_calib(fsweep_requested[0]), bad_amp_calib(fsweep_requested[1]))
amp_bounds = (better_amp_calib(fsweep_requested[0]), better_amp_calib(fsweep_requested[1]))
fsweep_actual = (poly_freq_calib(amp_bounds[0]), poly_freq_calib(amp_bounds[1]))
rabi_range = abs(fsweep_actual[1] - fsweep_actual[0])

# apply butterworth filter (same as rabi_sweep)
filter_b, filter_a = signal.butter(8, 2 * filter_cutoff / lockin_sample_rate)
filter_Q = signal.filtfilt(filter_b, filter_a, lockin_Q)

# Repeat Rabi sweep frequency calibrations for lockin trace:
amp_map = np.vectorize(amp_map)
rf_amps_lockin = amp_map(t_lockin, t0, rabiT, amp_bounds[0], amp_bounds[1])
rabi_freqs_lockin = poly_freq_calib(rf_amps_lockin)
ac_res_time_lockin = t_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]
ac_res_rabi_freq = rabi_freqs_lockin[np.argmin(abs(rabi_freqs_lockin - ac_freq))]

# Restrict data to desired range for plotting
filter_Q_sub = filter_Q[np.logical_and(rabi_freqs_lockin > ac_res_rabi_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_res_rabi_freq + rabi_range*subRange/2)]
t_lockin_sub = t_lockin[np.logical_and(rabi_freqs_lockin > ac_res_rabi_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_res_rabi_freq + rabi_range*subRange/2)]
rabi_freqs_sub = poly_freq_calib(amp_map(t_lockin_sub, t0, rabiT, amp_bounds[0], amp_bounds[1]))

# Readjust times such that t_res = 0, for plotting overlay
t_adjust_sub = t_lockin_sub - ac_res_time_lockin

# Compute sweep bounds (time and freq) and predicted resonances from AC signals for lockin
bound_times_lockin = [t_lockin[np.argmin(abs(t_lockin - t_b))] for t_b in tsweep]
bound_freqs_lockin = [rabi_freqs_lockin[np.argmin(abs(t_lockin - t_b))] for t_b in tsweep]
# # set up secondary axis ticks
# rabi_tick_locs = [bound_times_lockin[0]]
# rabi_tick_freqs = ["%0.2f" % (bound_freqs_lockin[0]/1e3)]
# rabi_tick_locs.append(ac_res_time_lockin)
# rabi_tick_freqs.append("%0.2f" % (ac_res_rabi_freq/1e3))
# rabi_tick_locs.append(bound_times_lockin[1])
# rabi_tick_freqs.append("%0.2f" % (bound_freqs_lockin[1]/1e3))

#-----------------------------------------------------------------------------------------#
# Fit reconstruction
#-----------------------------------------------------------------------------------------#

# Import some additional data we need to convert amplitudes to guess Rabi_2 freqs
ampToGauss = globs['Bz0_G_per_V']

# Cut frequency region for fitting to be around the transition
# rabi_freqs_sub = rabi_freqs_lockin[np.logical_and(rabi_freqs_lockin > ac_res_rabi_freq - rabi_range*subRange/2, rabi_freqs_lockin < ac_res_rabi_freq + rabi_range*subRange/2)]

# try to reconstruct fits
print('Importing previous fit results for plotting')
if ('rabi_sweep_fitting','amp') in dataset:
	print('Adiabatic fit found in hdf5, reconstructing...')
	f_res = dataset[('rabi_sweep_fitting','f_res')]
	f_res_exp = dataset[('rabi_sweep_fitting','f_res_guess')]
	u_f_res = dataset[('rabi_sweep_fitting','u_f_res')]
	f_rabi2 = dataset[('rabi_sweep_fitting','f_rabi2')]
	f_rabi2_exp = dataset[('rabi_sweep_fitting','f_rabi2_guess')]
	u_f_rabi2 = dataset[('rabi_sweep_fitting','u_f_rabi2')]
	amp = dataset[('rabi_sweep_fitting','amp')]
	adiabaticFitRecon = adiabatic_spin_tip(f_rabi_1 = rabi_freqs_sub, f_res = f_res, f_rabi_2 = f_rabi2, amp = amp)
	result_string_ad1 = r'$f_{{2S}}$ = {:.3f}({:.0f}) kHz'.format(f_res/1e3, u_f_res)
	result_string_ad2 = r'$\Omega_{{2S}}$ = {:.0f}({:.0f}) Hz'.format(f_rabi2, u_f_rabi2)
	result_string_ad = result_string_ad1 + '\n' + result_string_ad2
	result_string_exp1 = r'$f_{{2A}}$ = {:.0f} kHz'.format(f_res_exp/1e3)
	result_string_exp2 = r'$\Omega_{{2A}}$ = {:.0f} Hz'.format(f_rabi2_exp)
	result_string_exp = result_string_exp1 + '\n' + result_string_exp2
else:
	adiabaticFitRecon = [False]
	result_string_ad = ''
if ('rabi_sweep_fitting','amp_analytic') in dataset:
	print('Analytic fit found in hdf5, reconstructing...')
	t_res_a = dataset[('rabi_sweep_fitting','t_res_analytic')]
	f_res_a = dataset[('rabi_sweep_fitting','f_res_analytic')]
	u_f_res_a = dataset[('rabi_sweep_fitting','u_f_res_analytic')]
	f_rabi2_a = dataset[('rabi_sweep_fitting','f_rabi2_analytic')]
	u_f_rabi2_a = dataset[('rabi_sweep_fitting','u_f_rabi2_analytic')]
	amp_a = dataset[('rabi_sweep_fitting','amp_analytic')]
	v_est_a = dataset[('rabi_sweep_fitting','v_est_analytic')]
	# analyticFitRecon = rhoVec(rabi_times = t_lockin_sub, t_res = t_res_a, f_rabi_2 = f_rabi2_a, amp = amp_a, v_est = v_est_a)
	analyticFitRecon = rhoVec_ang(rabi_times = t_lockin_sub, t_res = t_res_a, f_rabi_2 = f_rabi2_a, amp = amp_a, v_est = v_est_a)
	result_string_an1 = r'$f_{{2S}}$ = {:.4f}({:.0f}) kHz'.format(f_res_a/1e3, u_f_res_a*10)
	result_string_an2 = r'$\Omega_{{2S}}$ = {:.1f}({:.0f}) Hz'.format(f_rabi2_a, u_f_rabi2_a*10)
	result_string_an = result_string_an1 + '\n' + result_string_an2
else:
	analyticFitRecon = [False]
	result_string_an = ''
if len(adiabaticFitRecon) == 1 and len(analyticFitRecon) == 1:
	print('No fit results found in hdf5.')

#-----------------------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Define some plot limits
larmor = dataset[('faraday','larmor_guess')]
spec_ylims = np.array((larmor - np.max(rabi_freqs_sub), larmor + np.max(rabi_freqs_sub)))/1e3
demod_ylims = (-max(filter_Q_sub)*1e3-0.05, max(filter_Q_sub)*1e3+0.05)
time_lims = (np.min(t_lockin_sub), np.max(t_lockin_sub))
trange = time_lims[1] - time_lims[0]
Qrange = demod_ylims[1] - demod_ylims[0]
spectrogram_lims = [1, 3.5]

# Define figure object
fig = plt.figure('specQcompare', figsize = (10,6))

# Plot and label spectrogram
ax1 = fig.add_subplot(2,1,1)
# ax1.plot_stft(stft, t, f, range=spectrogram_lims, png=None)
ax1.set_title('Fitting for a non-adiabatic transition')
ax1.imshow(np.log10(stft.T), origin='lower', vmin=spectrogram_lims[0], vmax=spectrogram_lims[1], extent=[t[0], t[-1], f[0]/1e3, f[-1]/1e3], aspect='auto', interpolation='nearest', cmap='jet')
ax1.axvline(ac_res_time_lockin, c = 'm', ls = '--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [kHz]')
ax1.set_xlim(time_lims)
ax1.set_ylim(spec_ylims)

# Make second subplot for demod channel, w/ fits
ax2 = fig.add_subplot(2,1,2)
ax2.spines["top"].set_visible(False)
ax2.axhline(0, c = 'lightgray', alpha = 0.5)
ax2.plot(t_lockin_sub, filter_Q_sub*1e3, label = 'filtered Q(t)')
ax2.plot(t_lockin_sub, adiabaticFitRecon*1e3, ls = '--', label = 'adiabatic fit')
ax2.plot(t_lockin_sub, analyticFitRecon*1e3, ls = '--', c = 'k', label = 'analytic fit')
ax2.axvline(ac_res_time_lockin, c = 'm', ls = '--', label = r'$f_{{2A}}$ = {:.2f} kHz'.format(ac_res_rabi_freq/1e3))
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Q(t) [mV]')
ax2.set_ylim(demod_ylims)
ax2.set_xlim(time_lims)
ax2.legend()
plt.subplots_adjust(hspace = 0)

# Define ticks for Rabi frequency
rabi_tick_locs = ax2.get_xticks()
rabi_tick_freqs = np.around(poly_freq_calib(amp_map(rabi_tick_locs, t0, rabiT, amp_bounds[0], amp_bounds[1]))/1e3, 2)

# Make secondary axis for Rabi frequency on spectrogram
ax22 = ax1.twiny()
ax22.set_xticks(rabi_tick_locs)
ax22.set_xticklabels(rabi_tick_freqs)
ax22.set_xlim(ax2.get_xlim())
ax22.set_xlabel(r'$\Omega_{{1R}}$ [kHz]')
ax1.xaxis.set_visible(False)
ax1.spines["bottom"].set_visible(False)

# Convert text coords into positions based on plot ranges
def textCoords_x(coord, tmin=time_lims[0], trange=trange):
	return tmin + coord*trange
def textCoords_y(coord, Qmin=demod_ylims[0], Qrange=Qrange):
	return Qmin + coord*Qrange

# Add results text to plot
# ax2.text(0.21, 0.25, 'fit results:', c = colours[0])
# ax2.text(0.21, -0.1, result_string_ad, c = colours[1])
# ax2.text(0.21, -0.4, result_string_an, c = 'k')
# ax2.text(0.21, -0.7, result_string_exp, c = 'g')
ax2.text(textCoords_x(0.042), textCoords_y(0.587), 'fit results:', c = colours[0])
ax2.text(textCoords_x(0.042), textCoords_y(0.411), result_string_ad, c = colours[1])
ax2.text(textCoords_x(0.042), textCoords_y(0.245), result_string_an, c = 'k')
ax2.text(textCoords_x(0.042), textCoords_y(0.088), result_string_exp, c = 'g')
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
# ax2.add_patch(patches.Rectangle((textCoords_x(0.03), textCoords_y(0.0588)), 0.192*trange, 0.676*Qrange, facecolor = 'white', edgecolor = 'k', alpha = 0.5))
ax2.add_patch(patches.Rectangle((textCoords_x(0.03), textCoords_y(0.0588)), 0.205*trange, 0.62*Qrange, facecolor = 'lightgrey', edgecolor = 'k', alpha = 0.5))

plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'nonAdiabaticFits.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'nonAdiabaticFits.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))