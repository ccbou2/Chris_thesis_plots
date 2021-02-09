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

#-----------------------------------------------------------------------------------------#
# Import data
#-----------------------------------------------------------------------------------------#
# Load lyse dataset
df = data()

# Locate desired shots via last sequence or timestamp
# subdf = df.sequences(last=1) #.iloc[:-2]
# subdf = df.loc['20190925T161248']
try:
	subdf = df.loc[['20190924T112141','20190924T162441']]
	# subdf = df.loc[['20190924T112141','20190924T162441']]
except KeyError:
	print ('Cant locate shots 20190924T112141, 20190924T162441 in lyse.')
# seq_str = subdf.sequence_string()

# Re-index subdf based on variables we care about (amplitude and frequency for now)
# subdf.set_index([df.tuplify('magnetom_ac_amplitude'), df.tuplify('magnetom_ac_frequency')], inplace = True)

# # Retrieve data based on a set ac frequency (second index), all cols, set freq to desired selection
# freqdf = subdf.loc[(slice(None), 11000.0), :]

#-----------------------------------------------------------------------------------------#
# Process data
#-----------------------------------------------------------------------------------------#

path1 = subdf['filepath'][0]
path2 = subdf['filepath'][1]
print(path1, path2)
dataset1 = data(path1)
run1 = Run(path1)
dataset2 = data(path2)
run2 = Run(path2)
# shot_id = os.path.split(path)[-1].replace('.h5', '')

# Load spectrogram and lockin Q channel from hdf5
t1, f1, stft1, globs1 = load_stft(path1, all_globals=True)
t2, f2, stft2, globs2 = load_stft(path2, all_globals=True)
# t_lockin, lockin_Q, lockin_globs = get_lockin_trace(path, '/data/traces/lockin_connection/Y_channel')

if stft1 is None:
	pass
	print(Exception('No STFT present'))
else:
	# Analysis parameters
	f_sp1 = globs1['sp_center_freq']*1e6

#-----------------------------------------------------------------------------------------#
# Plot
#-----------------------------------------------------------------------------------------#

colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Define some plot limits
larmor1 = dataset1[('faraday','larmor_guess')]
# larmor2 = dataset2[('faraday','larmor_guess')]
larmor2 = 483000
spec_ylims1 = np.array((larmor1/1e3 - 10, larmor1/1e3 + 10))
spec_ylims2 = np.array((larmor2/1e3 - 10, larmor2/1e3 + 10))
# demod_ylims = (-max(filter_Q_sub)*1e3-0.05, max(filter_Q_sub)*1e3+0.05)
# demod_ylims = (-max(filter_Q)*1e3-0.05, max(filter_Q)*1e3+0.05)
time_lims = (np.min(t1), np.max(t1)-0.01)
# time_lims = (np.min(t_lockin), np.max(t_lockin))
# trange = time_lims[1] - time_lims[0]
# Qrange = demod_ylims[1] - demod_ylims[0]
spectrogram_lims = [1, 3.5]

# Define figure object
fig = plt.figure('dressed_vs_undressed', figsize = (10,6))

# Plot and label spectrogram (undressed)
ax1 = fig.add_subplot(2,1,1)
# ax1.plot_stft(stft, t, f, range=spectrogram_lims, png=None)
ax1.set_title('Dressed vs Undressed Faraday spectrograms')
ax1.imshow(np.log10(stft1.T), origin='lower', vmin=spectrogram_lims[0], vmax=spectrogram_lims[1], extent=[t1[0], t1[-1], f1[0]/1e3, f1[-1]/1e3], aspect='auto', interpolation='nearest', cmap='jet')
# for res_time in ac_res_time_lockins:
# 	ax1.axvline(res_time, c = 'm', ls = '--')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Frequency [kHz]')
ax1.set_xlim(time_lims)
ax1.set_ylim(spec_ylims1)
ax1.xaxis.set_visible(False)
# ax1.spines["bottom"].set_visible(False)

# Plot and label second spectrogram (dressed)
ax2 = fig.add_subplot(2,1,2)
ax2.imshow(np.log10(stft2.T), origin='lower', vmin=spectrogram_lims[0], vmax=spectrogram_lims[1], extent=[t2[0], t2[-1], f2[0]/1e3, f2[-1]/1e3], aspect='auto', interpolation='nearest', cmap='jet')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Frequency [kHz]')
ax2.set_xlim(time_lims)
ax2.set_ylim(spec_ylims2)
plt.subplots_adjust(hspace = 0.1)

plt.show()

# Save figures
plt.savefig(os.path.join(folder, 'dressed_vs_undressed_spec.png'), dpi = 300, bbox_inches='tight')
plt.savefig(os.path.join(folder, 'dressed_vs_undressed_spec.pdf'), dpi = 300, bbox_inches='tight')
print('Plotting complete, outputs saved to ' + str(folder))