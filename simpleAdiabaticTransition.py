#!python3

from pylab import *
import numpy as np

def adiabatic_spin_tip(f_rabi_1, f_res, f_rabi_2, amp):
	# amp factor to convert to recorded voltages, and allow for both directions of spin tip
	return amp*(f_rabi_1 - f_res)/sqrt((f_rabi_1 - f_res)**2 + f_rabi_2**2)

# detunings = linspace(-15000, 15000, 1000)
# plusState = 0.1 * detunings
# minusState = -0.1 * detunings
# plus2State = sqrt(300**2 + plusState**2)
# minus2State = -sqrt(300**2 + minusState**2)
# horizontalAxis = zeros(len(detunings))

# times = linspace(-10, 10, 1000)
rabiFreqs = linspace(0, 20, 1000)
spin_tip = adiabatic_spin_tip(rabiFreqs, 10, 1, -1)
horizontalAxis = zeros(len(rabiFreqs))

figure(figsize = (12,8))
plot(rabiFreqs, spin_tip)
# plot(detunings, minusState, c = 'slategrey', ls = '--')
# plot(detunings, plus2State)
# plot(detunings, minus2State)
plot(rabiFreqs, horizontalAxis, c = 'k', ls = '-')
axvline(0, ymin=min(spin_tip), ymax=max(spin_tip), c='k', ls='-')
xlim(0, 20)
ylim(-1, 1)
frame = gca()
# frame.axis('off')
frame.set_xlabel(r'$\langle F_z \rangle$')
frame.axes.get_xaxis().set_ticklabels([])
frame.axes.get_yaxis().set_ticklabels([])
frame.xaxis.set_label_position('top')
frame.yaxis.set_label_position('right')
frame.set_ylabel(r'$\Delta$')
frame
# h = ylabel('E')
# h.set_rotation(0)
grid()
show()