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
with open("BPSK.txt") as fileobject:
    l = 0
    for line in fileobject:
        reading = line[0: line.find(",")]
        raw_samples.append(complex(reading))
        l=+1
fileobject.close()
#print(raw_samples)

#################Read From Textfile##################



phase_raw_samples = np.unwrap(np.angle(raw_samples))*-1
#phase_raw_samples = sc.signal.savgol_filter(phase_raw_samples1, 11, 5)
pll_out = np.zeros([30000], dtype=complex)
new_phase_raw_samples = np.zeros([30000], dtype=complex)
n=0
m=0
for i in range(29999):
    n=n+phase_raw_samples[i]
    m+=1
    if ((phase_raw_samples[i]-phase_raw_samples[i-1])>=3*np.pi/2):
        for j in range(m):
            new_phase_raw_samples[i-j] = n/m
        m=0
        n=0


counter = 0
m = 0
plekhouer = 0
flaggie = 0

for i in range(len(phase_raw_samples)-1):
    print(phase_raw_samples[i])
    if flaggie == 0:
        if (phase_raw_samples[i] > 150):
            if (counter < 50):
                print("####################")
                print(phase_raw_samples[i])
                #m = (m - phase_raw_samples[i])/2
                counter = counter + 1
            elif counter >= 50:
                m = (phase_raw_samples[i] - phase_raw_samples[i-30])/30
                plekhouer = i - 50
                flaggie = 1
        else:
            counter =0

#m = 0.2#m/50
for i in range(len(pll_out)-1):
    pll_out[i] = phase_raw_samples[i]
    if i > plekhouer:
        pll_out[i] = m + pll_out[i-1]


modulation_out = raw_samples * np.exp(-1j * pll_out)

#PLL phase
plt.figure(1)
#plt.subplot(221)
plt.plot(new_phase_raw_samples, label='phase_raw_samples')
#plt.plot(pll_out, label='PLL')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('samples')
plt.ylabel('Phase')
plt.grid(True)

plt.figure(2)
#plt.subplot(221)
plt.plot(modulation_out, label='modulation_out')
plt.legend()
plt.yscale('linear')
plt.title('phase_diff')
plt.xlabel('samples')
plt.ylabel('Phase')
plt.grid(True)


plt.figure(3)
#plt.subplot(221)
plt.plot(np.real(modulation_out), np.imag(modulation_out), 'bo', label='modulation_out IQ')
plt.legend()
plt.yscale('linear')
plt.title('phase_diff')
plt.xlabel('real')
plt.ylabel('imag')
plt.grid(True)
'''

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
'''
plt.show()
