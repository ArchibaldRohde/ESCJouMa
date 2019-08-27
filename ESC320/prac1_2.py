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

#Read CSV file

with open('bursts_weirdness.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    l = 0
    for row in csv_reader:
        piele = row[0]
        piele = piele.replace("i", "j")
        samples0.append(complex(piele))
        raw_samples.append(complex(piele))
        magnitude_samples.append(abs(raw_samples[l]))
        #abs_raw_samples.append()
        l += 1
    csv_file.close()


#vars
phase_samples_wrap = np.angle(raw_samples)                  #phases from -pi to pi
phase_samples_accum = np.abs(np.unwrap(phase_samples_wrap)) #accumulating phases

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
'''

for i in phase_samples_accum:
    e = i - predictor[j-1]
    d = d + predictor[j-1] + i
    predictor[j] = 0.9*d + 1.1*e
    # print(i)
    j += 1
'''

####Savitzy-goley

savgol_array = sc.signal.savgol_filter(predictor, 101, 3)

PLL_out_phases = np.delete(PLL_out_phases, 30000)

fourier = []
fmag = []

for i in range(len(savgol_array) - 1):
    PLL_out_phases[i] = (savgol_array[i] + np.pi) % (2 * np.pi) - np.pi #unwraped phases
    a[i] = np.cos(PLL_out_phases[i]) * magnitude_samples[i]
    print(a[i])
    b[i] = np.sin(PLL_out_phases[i]) * magnitude_samples[i]
    print(b[i])
    reconstructed_signal[i] = complex(a[i], b[i])
    print(reconstructed_signal[i])
    #fourier.append(np.sqrt(complex.real(raw_samples[i])**2+complex.imag(raw_samples[i])**2))
    fourier.append(np.abs(raw_samples[i]))#*np.conj(raw_samples[i])))
    fmag.append(20 * np.log(fourier[i]))



#fmag = sc.fftpack.fft(fourier, 2.4e6)
#fmag = 20*np.log(fmag)

f_s = 2.4e6
X = sc.fft(fmag)
freqs = sc.fftpack.fftfreq(len(fmag)) * f_s

fig, ax = plt.subplots()
ax.stem(freqs, np.abs(X), use_line_collection=True)
ax.stem(freqs, fmag, use_line_collection=True)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
#ax.set_xlim(-f_s/2, f_s/2)
plot(X)

plt.show()