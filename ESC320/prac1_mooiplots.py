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
#from scipy import fftpack

#for g in range(11):


samples0 = []
raw_samples = []
magnitude_samples = []
abs_raw_samples = []

#inputs = np.genfromtxt('.txt', dtype=complex)

with open('bursts_weirdness.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    l = 0
    for row in csv_reader:
        piele = row[2]
        piele = piele.replace("i", "j")
        samples0.append(complex(piele))
        raw_samples.append(complex(piele))
        magnitude_samples.append(abs(raw_samples[l]))
        #abs_raw_samples.append()
        l += 1
    csv_file.close()

phase_samples_accum = np.abs(np.unwrap(np.angle(raw_samples)))
phase_samples_wrap = np.angle(raw_samples)

PLL_out_phases = np.zeros([30001], dtype=float)
a = np.zeros([30001], dtype=float)
b = np.zeros([30001], dtype=float)
reconstructed_signal = np.zeros([30001], dtype=complex)
predictor = np.zeros([30001], dtype=float)
phase_diff_e = np.zeros([30001], dtype=float)
phase_diff = np.zeros([30001], dtype=float)
e = 0
k = 0
d = 0
j = 0

for i in phase_samples_accum:
    e = i - predictor[j]
    d = i - phase_samples_accum[j - 1]
    predictor[j + 1] = predictor[j] + e + d
    # print(i)
    phase_diff_e[j] = e
    j += 1

"""
for i in range(len(samples_unpacked)):
    if i < 30000:
        k += 1
        e = samples_unpacked[i] + e
        predictor[i] = d
        if k == 10:
            d = e/10
            predictor[i] = d
            #print(i)
            k = 0
            e = 0
"""

####Savitzy-goley

savgol_array = sc.signal.savgol_filter(predictor, 101, 3)

PLL_out_phases = np.delete(PLL_out_phases, 30000)

print(len(PLL_out_phases))
print('PLL_out_phases')
print(len(magnitude_samples))
print('magnitude_samples')
fourier = []
fmag = []

for i in range(len(savgol_array) - 1):
    PLL_out_phases[i] = (savgol_array[i] + np.pi) % (2 * np.pi) - np.pi
    a[i] = np.cos(PLL_out_phases[i]) * magnitude_samples[i]
    b[i] = np.sin(PLL_out_phases[i]) * magnitude_samples[i]
    reconstructed_signal[i] = complex(a[i], b[i])
    #fourier.append(np.sqrt(complex.real(raw_samples[i])**2+complex.imag(raw_samples[i])**2))
    fourier.append(raw_samples[i])#*np.conj(raw_samples[i])))
    #fmag.append(10 * np.log(fourier[i]))

results_frequency = np.gradient(savgol_array)#savgol_array)
samples_frequency = np.gradient(phase_samples_accum)


# samples_unwrap = complex(np.cos(predictor), np.sin(predictor))


#plt.xlabel('samples')
#plt.ylabel('Amplificaions')

#fmag = sc.fftpack.fft(fourier, 2.4e6)
#fmag = 20*np.log(fmag)

f_s = 2.4e6

Y = np.abs(fourier[10000:20000])
X = np.abs(sc.fft(Y))
fmag = 20 * np.log(X/max(X))

freqs = sc.fftpack.fftfreq(len(fmag)) *f_s#/len(fmag)

fig, ax = plt.subplots()
ax.stem(freqs, np.abs(X), use_line_collection=True)
#ax.stem(freqs, fmag, use_line_collection=True)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s +30000, f_s +30000)
#ax.set_ylim(-5, 110)


# plot with various axes scales
plt.figure(1)

##Phase
#plot(phase_samples_accum)
#plot(savgol_array)
plt.subplot(221)
plt.plot(phase_samples_accum, label='samples')
plt.plot(savgol_array, label='PLL')
plt.legend()
plt.yscale('linear')
plt.title('phase PLL')
plt.xlabel('samples')
plt.ylabel('phase')
plt.grid(True)

##Frequency
#plot(samples_frequency)
#plot(results_frequency)
plt.subplot(222)
plt.plot(samples_frequency, label='samples')
plt.plot(results_frequency, label='results')
plt.legend()
plt.yscale('linear')
plt.title('Frequency')
plt.xlabel('samples')
plt.ylabel('phase')
plt.grid(True)


##sinus tyd
#plot(raw_samples)
#plot(reconstructed_signal)
jas = raw_samples/np.exp(1j*PLL_out_phases)
plt.subplot(223)
plt.plot(jas, label='PLL_out_phases')
plt.legend()
plt.yscale('linear')
plt.title('PLL_out_phases')
plt.xlabel('samples')
plt.ylabel('phase')
plt.grid(True)
'''
plt.plot(raw_samples, label='raw samples')
plt.plot(reconstructed_signal, label='reconstructed_signal')
plt.legend()
plt.yscale('linear')
plt.title('reconstructed signal')
plt.xlabel('samples')
plt.ylabel('phase')
plt.grid(True)
'''

savgol_array = savgol_array[0:30000]
plt.subplot(224)
plt.plot((phase_samples_accum - savgol_array), label='phase difference')
#plt.plot((phase_samples_wrap - PLL_out_phases), label='phase difference')
plt.legend()
plt.yscale('linear')
plt.title('phase difference')
plt.xlabel('samples')
plt.ylabel('phase')
plt.grid(True)


'''
# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)
'''


#plot(fourier)
#plt.subplot(234)
plt.figure(2)

plt.plot(freqs, fmag, label='fourier transform')
plt.legend()
plt.yscale('linear')
plt.title('fourier transform')
plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.82, bottom=0.18, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()