import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.pyplot import *
import pylab as py
from rtlsdr import *
import csv
import cmath
import plotly.graph_objects as go
import pandas as pd
import scipy as sc#
from scipy import signal
from scipy.signal import butter, lfilter, freqz
#from scipy import fftpack

#for g in range(11):
#leerman = np.zeros([30000], dtype=str)
#raw_samples = np.zeros([30000], dtype=complex)

#################Read From Textfile##################

raw_samples = []
with open("16QAM.txt") as fileobject:
    l = 0
    for line in fileobject:
        reading = line[0: line.find(",")]
        raw_samples.append(complex(reading))
        l=+1
fileobject.close()
print(raw_samples)

#################Read From Textfile##################



phase_raw_samples = np.unwrap(np.angle(raw_samples))*-1
print(phase_raw_samples[6000])
derivative_raw_samples = np.zeros([30000], dtype=complex)
integral = np.zeros([30000], dtype=complex)
LPF_out_phase = np.zeros([30000], dtype=complex)
LPF_out = np.zeros([30000], dtype=complex)

for i in range(29999):
    derivative_raw_samples[i] = (phase_raw_samples[i+1]-phase_raw_samples[i])####<-----------------------------------
    #derivative_raw_samples[i] = 0.22

    temp = 0
    if i > 100:
        for j in range(40):
            temp = temp + derivative_raw_samples[i-j]
        LPF_out[i] = (temp/40)
        #LPF_out_phase[i] = ((temp/50 + np.pi) % (2 * np.pi) - np.pi)
        '''#LPF_out.append((derivative_raw_samples[i-10]+derivative_raw_samples[i-9]+derivative_raw_samples[i-8]
           #             +derivative_raw_samples[i-7]+derivative_raw_samples[i-6]+derivative_raw_samples[i-5]
           #             +derivative_raw_samples[i-4]+derivative_raw_samples[i-3]+derivative_raw_samples[i-2]
            #            +derivative_raw_samples[i-1])/10)
'''
        integral[i] = LPF_out[i]

for k in range(len(integral)-1):
    integral[k+1] = integral[k+1] + integral[k]


print(phase_raw_samples)
print(derivative_raw_samples)
print(LPF_out)

phase_difference = np.zeros([30000], dtype=complex)

for h in range(len(phase_raw_samples)):
    phase_difference[h] = phase_raw_samples[h] - integral[h]

phase_difference = np.angle(np.exp(1j*phase_difference))
#for g in range(len(phase_raw_samples)):
#    phase_difference[g] = ((phase_difference[g] + np.pi) % (2 * np.pi) - np.pi)


#PLL phase
plt.figure(1)
#plt.subplot(221)
plt.plot(phase_raw_samples, label='phase_raw_samples')
plt.plot(integral, label='PLL')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('samples')
plt.ylabel('Phase')
plt.grid(True)

#Frequency
#plt.subplot(222)
plt.figure(2)
plt.plot(derivative_raw_samples, label='derivative_raw_samples')
plt.plot(LPF_out, label='LPF')
plt.legend()
plt.yscale('linear')
plt.title('frequency PLL')
plt.xlabel('samples')
plt.ylabel('Frequency')
plt.grid(True)

plt.figure(3)
#plt.subplot(223)
plt.plot(phase_difference, label='phase diff')
plt.legend()
plt.yscale('linear')
plt.title('Phase difference')
plt.xlabel('samples')
plt.ylabel('Phase difference')
plt.grid(True)


jas = np.exp(1j*phase_difference)
plt.figure(10)
#plt.subplot(223)
plt.plot(np.real(jas), np.imag(jas), 'bo', label='phase diff')
plt.legend()
plt.yscale('linear')
plt.title('Phase difference')
plt.xlabel('samples')
plt.ylabel('Phase difference')
plt.grid(True)



plt.figure(4)

for s in range(29999):
    if ((phase_difference[s]<np.pi) and (phase_difference[s]>np.pi/2)):
        phase_difference[s] = np.pi
    if ((phase_difference[s]<np.pi/2) and (phase_difference[s]>0)):
        phase_difference[s] = np.pi/2
    if ((phase_difference[s]<0) and (phase_difference[s]>(-np.pi/2))):
        phase_difference[s] = -np.pi/2
    if ((phase_difference[s]<(-np.pi/2)) and (phase_difference[s]>(-np.pi))):
        phase_difference[s] = -np.pi

plt.plot(phase_difference, label='phase diff')
#(x+jy)/exp(j*pll)
plt.legend()
plt.yscale('linear')
plt.title('Phase difference')
plt.xlabel('samples')
plt.ylabel('Phase difference')
plt.grid(True)

#plot(fourier)
#plt.subplot(234)

f_s = 2.4e6

Y = np.abs(raw_samples[1000:29000])
X = np.abs(sc.fft(Y,300000))
fmag1 = 10 * np.log(X/max(X))
fmag2 = 20 * np.log(X/max(X))

freqs = sc.fftpack.fftfreq(len(fmag1)) *f_s#/len(fmag)
plt.figure(5)

fig, ax = plt.subplots()
#ax.yscale('linear')
#plt.title('Fourier transform')
#ax.ylabel('frequency[Hz]')
#ax.ylabel('Amplitude[dB]')
#ax.legend()
ax.plot(freqs, fmag2, label='fourier transform')
#plt.plot(freqs, fmag1, label='fourier transform')

ax.grid(True)
plt.show()

'''
###########################################
# Copied this from the internet
# https://docs.python.org/3/library/csv.html
with open('rtslr_out1.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    l = 0
    for row in csv_reader:
        print(row[0])
        print("#################################")
        leerman = row[0]
        print(leerman)
        leerman1 = leerman.replace("i", "j")
        print(leerman1)
        #samples0.append(complex(piele))
        #if l%2 == 0:
        raw_samples[l] = (complex(leerman1))
        print(raw_samples[l])
        #magnitude_samples.append(abs(raw_samples[l]))
        #abs_raw_samples.append()
        l += 1
    csv_file.close()
##################################################
'''


'''
#############################################################################
#https://matplotlib.org/3.1.1/gallery/misc/cursor_demo_sgskip.html
class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3)
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x],[y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x,y))
        self.ax.figure.canvas.draw_idle()
###############################################################################
'''