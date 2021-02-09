#!python3
from __future__ import division
from lyse import *
from pylab import *
import numpy as np
from analysislib.common import *
from analysislib.spinor.aliases import *
from analysislib.spinor.faraday.faraday_aux import *
# from analysislib.spinor.faraday.faraday_demod import get_lockin_trace
# from analysislib.spinor.faraday.rabi_sweep_fitting import bad_amp_calib, amp_map
from scipy import signal, special
import labscript_utils.h5_lock
import h5py
import lmfit
import pandas as pd

# New
def import_globs(path, trace_location='/data/traces/alazar_faraday/channelA'):
    # extract the raw Hobbs measurement from the file
    # as acquired by the Alazar ATS9462 card
    with h5py.File(path,'r') as F:
        # read out the globals dict
        globs = dict(F['/globals'].attrs)
    return globs

# New
def fit_filter_nans(xList, yList, u_yList = None):
	# Take x and y list (u_y list optional) and filter y list for nans
	tempx = []
	tempy = []
	tempuy = []
	for i in range(len(yList)):
		if not np.isnan(yList[i]):
			tempx.append(xList[i])
			tempy.append(yList[i])
			if u_yList is not None:
				tempuy.append(u_yList[i])
	if u_yList is not None:
		return np.array(tempx), np.array(tempy), np.array(tempuy)
	else:
		return np.array(tempx), np.array(tempy)


# rabi_sweep_fitting
def bad_amp_calib(f):
	# Use for timestamps before 20191008T152137
	return (f + 400)/17000

# rabi_sweep_fitting
def better_amp_calib(f):
	# Use for timestamps 20191008T153643 onwards
	return (f + 650)/17000

# new
def amp_calib(f, shot_id):
	if int(shot_id[:8]) == 20191008 and int(shot_id[9:15]) > 153600:
		# print('Using "better" linear calibration to determine rf_coil amp bounds from shot')
		return better_amp_calib(f)
	else:
		# print('Using original "bad" linear calibration to determine rf_coil amp bounds from shot')
		return bad_amp_calib(f)

# rabi_sweep_fitting
def poly_freq_calib(amp):
	# if 'rabi_calib_grad' in globs:
	#     # Load coded globals if tstamp suitable to contain them
	#     ml = globs['rabi_calib_lin_grad']
	#     mq = globs['rabi_calib_quad_grad']
	#     mc = globs['rabi_calib_cube_grad']
	#     c = globs['rabi_calib_offset']
	# else:
	#     # hard-coded calibrated fit factors from Rabi fit in log 20191008:
	ml = 9671.49
	u_ml = 387.63
	mq = 11480.75
	u_mq = 763.12
	mc = -5232.11
	u_mc = 464.87
	c = 573.93
	u_c = 59.82
	return ml * amp + mq * amp**2 + mc * amp**3 + c

# rabi_sweep_fitting
def amp_map(t, t0, T, amp_init, amp_final):
	if t < t0:
		return amp_init
	if t > t0 + T:
		return amp_final
	# Returns a linear ramp of amplitude within sweep bounds
	return amp_init + (t - t0) * (amp_final-amp_init) / T

# rabi_sweep_fitting
def adiabatic_spin_tip(f_rabi_1, f_res, f_rabi_2, amp):
	# amp factor to convert to recorded voltages, and allow for both directions of spin tip
	return amp*(f_rabi_1 - f_res)/sqrt((f_rabi_1 - f_res)**2 + f_rabi_2**2)

# faraday_demod
def get_lockin_trace(path, trace_location='/data/traces/lockin_connection/Y_channel'):
    with h5py.File(path,'r') as F:
        params = dict(F[trace_location].attrs)
        R = params['actual capture sample rate (Hz)']
        V = F[trace_location].value
    print('Loaded {:d} points = {:.3f} s @ {:.2f} kSPS'.format(len(V), len(V)/R, R/1e3))
    t = np.arange(len(V))/R
    return t, V, params