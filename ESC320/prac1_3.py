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


samples0 = []
raw_samples = []
magnitude_samples = []
abs_raw_samples = []

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
###########################################Copied this from the internet
with open('bursts_weirdness.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    l = 0
    for row in csv_reader:
        piele = row[1]
        piele = piele.replace("i", "j")
        #samples0.append(complex(piele))
        raw_samples.append(complex(piele))
        #magnitude_samples.append(abs(raw_samples[l]))
        #abs_raw_samples.append()
        l += 1
    csv_file.close()
##################################################
phase_raw_samples = np.unwrap(np.angle(raw_samples))*-1
derivative_raw_samples = np.zeros([30000], dtype=complex)
integral = np.zeros([30000], dtype=complex)
LPF_out_phase = np.zeros([30000], dtype=complex)
LPF_out = np.zeros([30000], dtype=complex)

for i in range(29999):
    derivative_raw_samples[i] = (phase_raw_samples[i+1]-phase_raw_samples[i])
    temp = 0
    if i > 50:
        for j in range(50):
            temp = temp + derivative_raw_samples[i-j]
        LPF_out[i] = (temp/50)
        #LPF_out_phase[i] = ((temp/50 + np.pi) % (2 * np.pi) - np.pi)
        '''LPF_out.append((derivative_raw_samples[i-10]+derivative_raw_samples[i-9]+derivative_raw_samples[i-8]
                        +derivative_raw_samples[i-7]+derivative_raw_samples[i-6]+derivative_raw_samples[i-5]
                        +derivative_raw_samples[i-4]+derivative_raw_samples[i-3]+derivative_raw_samples[i-2]
                        +derivative_raw_samples[i-1])/10)
        '''
        integral[i] = LPF_out[i]

for k in range(len(integral)-1):
    integral[k+1] = integral[k+1] + integral[k]

print(phase_raw_samples)
print(derivative_raw_samples)
print(LPF_out)
'''
phase_difference = np.zeros([30000], dtype=complex)

for h in range(len(phase_raw_samples)):
    phase_difference[h] = phase_raw_samples[h] - integral[h]
for g in range(len(phase_raw_samples)):
    phase_difference[g] = ((phase_difference[g] + np.pi) % (2 * np.pi) - np.pi)
'''
plt.figure(1)
plt.plot((derivative_raw_samples - LPF_out_phase), label='phase diff')

#PLL phase
#plt.plot(phase_raw_samples, label='phase_raw_samples')
#plt.plot(integral, label='PLL')
#Frequency
#plt.plot(derivative_raw_samples, label='derivative_raw_samples')
#plt.plot(LPF_out, label='LPF')
plt.legend()
plt.yscale('linear')
plt.title('frequency PLL')
plt.xlabel('samples')
plt.ylabel('f')
plt.grid(True)

plt.show()