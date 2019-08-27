import numpy as np
import matplotlib.pyplot as plt
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



samples0 = []
raw_samples = []
magnitude_samples = []
abs_raw_samples = []

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
e = 0
k = 0
d = 0
j = 0

for i in phase_samples_accum:
    e = i - predictor[j]
    d = i - phase_samples_accum[j - 1]
    predictor[j + 1] = predictor[j] + e + d
    # print(i)
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
    fourier.append(np.real(raw_samples[i])**2)#*np.conj(raw_samples[i])))
    fmag.append(10 * np.log(fourier[i]))

results_frequency = np.diff(savgol_array)
samples_frequency = np.diff(raw_samples)



# samples_unwrap = complex(np.cos(predictor), np.sin(predictor))


#plt.xlabel('samples')
#plt.ylabel('Amplificaions')

#fmag = sc.fftpack.fft(fourier, 2.4e6)
#fmag = 20*np.log(fmag)

f_s = 2.4e6
X = sc.fftpack.fft(fmag)
freqs = sc.fftpack.fftfreq(len(fmag)) * f_s

fig, ax = plt.subplots()
ax.stem(freqs, np.abs(X), use_line_collection=True)
#ax.stem(freqs, fmag, use_line_collection=True)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s +30000, f_s +30000)
#ax.set_ylim(-5, 110)

##sinus tyd
#plot(raw_samples)
#plot(reconstructed_signal)

##Phase
#plot(phase_samples_accum)
#plot(savgol_array)

##Frequency
#plot(samples_frequency)
#plot(results_frequency)

##sample rate
#plot(fourier)

plt.show()
