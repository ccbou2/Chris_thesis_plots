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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
subdf = df.loc['20191002T124617']
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
filter_cutoff = 1e3
t0 = globs['hobbs_settle'] + globs['magnetom_ac_wait']
# rabiT = globs['magnetom_rabi_sweep_duration']
rabiT = globs['magnetom_ac_duration']
tsweep = (t0, t0 + rabiT)
# fsweep_requested = (globs['magnetom_rabi_sweep_initial'], globs['magnetom_rabi_sweep_final'])

# Signal params
ac_amp = globs['magnetom_ac_amplitude']
ac_freq = globs['magnetom_ac_frequency']

# apply butterworth filter (same as rabi_sweep)
filter_b, filter_a = signal.butter(8, 2 * filter_cutoff / lockin_sample_rate)
filter_Q = signal.filtfilt(filter_b, filter_a, lockin_Q)

# Compute sweep bounds (time and freq) and predicted resonances from AC signals for lockin
bound_times_lockin = [t_lockin[np.argmin(abs(t_lockin - t_b))] for t_b in tsweep]

#-----------------------------------------------------------------------------------------#
# Sideband LZ fitting
#-----------------------------------------------------------------------------------------#

# Redefine for simplicity the load_stft options
res = stft
t1 = t
f1 = f

# Import relevant globals
f_center = globs['sp_center_freq']*1e6
f_mag_center = globs['magnetom_ac_frequency']
f_mag_bandwidth = globs['magnetom_ac_sweep_bandwidth']
mag_t_start = globs['hobbs_settle'] + globs['magnetom_ac_wait']
mag_t_finish = mag_t_start + globs['magnetom_ac_duration']

def find_peaks(f, y, absthreshold=40000, min_dist=10):
    indexes = peakutils.indexes(y, thres=absthreshold/max(y), min_dist=min_dist)
    peaks = peakutils.interpolate(f, y, ind=indexes, width=min_dist)
    return [p for p in peaks if p >= f.min() and p <= f.max()]

# Isolate desired sideband data
# res_smaller = res[t1 > mag_t_start and t1 < mag_t_finish, f1 > 483e3 and f1 < 485e3]
res_smaller = res[np.logical_and(t1 > mag_t_start, t1 < mag_t_finish), :]
res_smaller = res_smaller[:, np.logical_and(f1 > 483e3, f1 < 485e3)]
# t_upper = t1[t1 > mag_t_start and t1 < mag_t_finish]
t_upper = t1[np.logical_and(t1 > mag_t_start, t1 < mag_t_finish)]
# t_upper = t1[t1 > mag_t_start]
# t_upper = t_upper[t_upper < mag_t_finish]
# f_upper = f1[f1 > 483e3 and f1 < 485e3]
f_upper = f1[np.logical_and(f1 > 483e3, f1 < 485e3)]
# f_upper = f1[f1 > 483e3]
# f_upper = f_upper[f_upper < 485e3]
f_peaks = f_upper[np.argmax(res_smaller, axis = 1)]

# for i in range(len(f1)):
#     if f1[i] > 483e3 and f1[i] < 485e3:
#         for j in range(len(t1)):
#             if t1[j] > mag_t_start and t1[j] < mag_t_finish:
#                 if np.log(res[j,f1 > 483e3:f1 < 485e3]) > 2:
#                     f_upper.append(f1[i])
#                     t_upper.append(t1[j])

# # Convert resulting arrays to numpy
# f_upper = np.array(f_upper)
# t_upper = np.array(t_upper)

# Convert times on x axis into detuned frequencies, and subtract offset from FFT freq
f_sweep_init = f_mag_center - f_mag_bandwidth/2
fi = (t_upper - mag_t_start)/globs['magnetom_ac_duration'] * f_mag_bandwidth + f_sweep_init
f_peaks -= f_center

# Define conversion function from time to frequency
def time_to_freq(t):
	return (t - mag_t_start)/globs['magnetom_ac_duration'] * f_mag_bandwidth + f_sweep_init

# Perform averages in intervals of 5 to reduce numbers of points
# f_peaks_red = np.mean(f_peaks[:(len(f_peaks)//11)*11].reshape(-1,11), axis=1)
# fi_red = fi[5::11]

# Define fit function
def rabiSpec(f_i, rabi, f_0):
    return np.sqrt(rabi**2 + (f_0 - f_i)**2)

# Define fit function
def rabiSpecLin(f_i, rabi, f_0, grad, f_c):
    return np.sqrt(rabi**2 + ((f_0 - f_i)*(grad + 1))**2)
# # Define fit function v2
# def rabiSpecLin(f_i, rabi, f_0, grad, f_c):
#     return np.sqrt(rabi**2 + (f_0*(1 + grad) - f_i * (1 + grad))**2)

# Set up fit with guess params and perform fit
model = lmfit.Model(rabiSpec)
model_params = model.make_params(rabi = 1500, f_0 = f_mag_center)
result = model.fit(f_peaks, model_params, f_i = fi)

# # Set up fit with extra linear grad underlying
# model2 = lmfit.Model(rabiSpecLin)
# model2_params = model2.make_params(rabi = 1500, f_0 = f_mag_center, grad = 1, f_c = f_mag_center)
# result2 = model2.fit(f_peaks, model2_params, f_i = fi)

# Output fit report
# print(result.fit_report())
# print(result2.fit_report())

# Retrieve fit params to paste:
sig_freq_str = r'$\Omega_2$ = {:.3f}({:.0f}) kHz'.format(result.params['rabi'].value/1e3, result.params['rabi'].stderr)
rabi_freq_str = r'$\Omega_R$ = {:.3f}({:.0f}) kHz'.format(result.params['f_0'].value/1e3, result.params['f_0'].stderr)
res_str = sig_freq_str + '\n' + rabi_freq_str

# # Plot fit result
# plt.figure('rabiSpectroscopy', figsize=(10,8))
# plt.plot(fi, f_peaks, linestyle = 'none', marker = 'o', label = 'data')
# # plt.plot(fi_red, f_peaks_red, 'bo', label = 'data')
# plt.plot(fi, result.best_fit, linestyle = 'solid', linewidth = 4.0, label = 'fit')
# # plt.plot(fi, result2.best_fit, 'r-', linewidth = 4.0, label = 'fit w/ offset')
# plt.grid()
# # plt.title(seq_str + '\n Rabi spectroscopy: upper sideband fit')
# plt.title('Rabi spectroscopy: upper sideband fit')
# plt.ylabel('Sideband frequency (Hz)')
# plt.xlabel('AC Signal frequency (Hz)')
# plt.legend()
# plt.show()

#-----------------------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Define some plot limits
larmor = dataset[('sp_center_freq')]*1e6
# spec_ylims = np.array((larmor - np.max(rabi_freqs_sub), larmor + np.max(rabi_freqs_sub)))/1e3
spec_ylims = np.array((larmor - 10e3, larmor + 10e3))/1e3
# spec_ylims = np.array((larmor - np.max(rabi_freqs_lockin) - 4e3, larmor + np.max(rabi_freqs_lockin) + 4e3))/1e3
# demod_ylims = (-max(filter_Q_sub)*1e3-0.05, max(filter_Q_sub)*1e3+0.05)
demod_ylims = (-max(filter_Q)*1e3-0.05, max(filter_Q)*1e3+0.05)
# time_lims = (np.min(t_lockin_sub), np.max(t_lockin_sub))
time_lims = (np.min(t_upper), np.max(t_upper))
Q_time_lims = (np.min(t_lockin), np.max(t_lockin))
freq_lims = (np.min((f_peaks+f_center)/1e3), np.max((f_peaks+f_center)/1e3))
freqRange = freq_lims[1] - freq_lims[0]
trange = time_lims[1] - time_lims[0]
Qrange = demod_ylims[1] - demod_ylims[0]
spectrogram_lims = [1, 3.5]

# Define figure object
gs_top = plt.GridSpec(3,1, hspace = 0)
gs_bottom = plt.GridSpec(3,1, hspace = 0.5)
fig = plt.figure('single_tone_LZ', figsize = (10,9))

# Plot and label spectrogram
ax1 = fig.add_subplot(gs_top[0,:])
# ax1.plot_stft(stft, t, f, range=spectrogram_lims, png=None)
ax1.set_title('Single-shot Rabi spectroscopy: frequency sweeps')
ax1.imshow(np.log10(stft.T), origin='lower', vmin=spectrogram_lims[0], vmax=spectrogram_lims[1], extent=[t[0], t[-1], f[0]/1e3, f[-1]/1e3], aspect='auto', interpolation='nearest', cmap='jet')
# ax1.axvline(ac_res_time_lockin, c = 'm', ls = '--')
ax1.axvline(bound_times_lockin[0], c = 'orange', ls = '--')
ax1.axvline(bound_times_lockin[1], c = 'orange', ls = '--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [kHz]')
ax1.set_xlim(Q_time_lims)
ax1.set_ylim(spec_ylims)
ax1.xaxis.set_visible(False)
ax1.spines["bottom"].set_visible(False)

# Make second subplot for demod channel, w/ fits
ax2 = fig.add_subplot(gs_top[1,:])
ax2.spines["top"].set_visible(False)
ax2.axhline(0, c = 'lightgray', alpha = 0.5)
ax2.plot(t_lockin, filter_Q*1e3, label = 'filtered Q(t)')
# ax2.plot(t_lockin_sub, adiabaticFitRecon*1e3, ls = '--', label = 'adiabatic fit')
# ax2.plot(t_lockin_sub, analyticFitRecon*1e3, ls = '--', c = 'k', label = 'analytic fit')
# ax2.axvline(ac_res_time_lockin, c = 'm', ls = '--', label = r'$f_s$ = {:.2f} kHz'.format(ac_res_rabi_freq/1e3))
ax2.axvline(bound_times_lockin[0], c = 'orange', ls = '--', label = 'Sweep bounds')
ax2.axvline(bound_times_lockin[1], c = 'orange', ls = '--')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Q(t) [mV]')
ax2.set_ylim(demod_ylims)
ax2.set_xlim(Q_time_lims)
ax2.legend()
plt.subplots_adjust(hspace = 0)

ax3 = fig.add_subplot(gs_bottom[2,:])
# ax3.plot(fi/1e3, f_peaks/1e3, linestyle = 'none', marker = 'o', label = 'data')
ax3.plot(t_upper, (f_peaks+f_center)/1e3, linestyle = 'none', marker = 'o', label = 'data')
# plt.plot(fi_red, f_peaks_red, 'bo', label = 'data')
# ax3.plot(fi/1e3, result.best_fit/1e3, linestyle = 'solid', linewidth = 4.0, label = 'fit')
ax3.plot(t_upper, (result.best_fit+f_center)/1e3, linestyle = 'solid', linewidth = 4.0, label = 'fit')
# plt.plot(fi, result2.best_fit, 'r-', linewidth = 4.0, label = 'fit w/ offset')
ax3.grid(color = 'lightgrey')
ax3.set_ylabel('Sideband frequency [kHz]')
ax3.set_yticklabels(['','1.5','2.0','2.5','3.0'])
ax3.set_xlabel('AC Signal frequency [kHz]')

ax3_times = ax3.get_xticks()
ax3_freqs = time_to_freq(ax3_times)/1e3
ax3.set_xticklabels(ax3_freqs)
# ax3.set_xlabel('Time [s]')

mark_inset(ax1, ax3, loc1 = 1, loc2 = 2, fc = 'crimson', ec = 'crimson', linestyle = 'dashdot', alpha = 0.4)

# Convert text coords into positions based on plot ranges
def textCoords_x(coord, tmin=time_lims[0], trange=trange):
	return tmin + coord*trange
def textCoords_y(coord, fmin=freq_lims[0], frange=freqRange):
	return fmin + coord*frange

# Add results text to plot
# ax2.text(0.21, 0.25, 'fit results:', c = colours[0])
# ax2.text(0.21, -0.1, result_string_ad, c = colours[1])
# ax2.text(0.21, -0.4, result_string_an, c = 'k')
# ax2.text(0.21, -0.7, result_string_exp, c = 'g')
ax3.text(textCoords_x(0.042 - 0.06), textCoords_y(0.345-0.05), 'fit results:', c = 'k')
ax3.text(textCoords_x(0.042 - 0.06), textCoords_y(0.088-0.05), res_str, c = colours[1])
# ax3.text(textCoords_x(0.042), textCoords_y(0.245), rabi_freq_str, c = colours[1])
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
ax3.add_patch(patches.Rectangle((textCoords_x(-0.03), textCoords_y(0.0588-0.05)), 0.205*trange, 0.4*freqRange, facecolor = 'lightgrey', edgecolor = 'k', alpha = 0.5))
ax3.legend()

plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'single_shot_Rabi.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'single_shot_Rabi.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))