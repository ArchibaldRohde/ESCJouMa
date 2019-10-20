import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.pyplot import *
import pylab as py
# from rtlsdr import *
import csv
import cmath
# import plotly.graph_objects as go
# import pandas as pd
import scipy as sc  #
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from commpy.filters import rrcosfilter

#################Read From Textfile##################

raw_samples = []
with open("RRC.txt") as fileobject:
    l = 0
    for line in fileobject:
        reading = line[0: line.find(",")]
        raw_samples.append(complex(reading))
        l = +1
fileobject.close()
# print(raw_samples)

#################Read From Textfile##################


# set carrier frequency
fsy = 100000
samplefreq = 2.4e6
fsymrate = 1.3e9  # + fsy

time, h_t = rrcosfilter(30000, 0.5, 8 / fsy, fsy)
# h_t = h_t/np.max(h_t)
# raw_samples = raw_samples/np.max(raw_samples)
jaaaa = signal.fftconvolve((raw_samples), h_t, mode='full')
# rawsquare = np.square(raw_samples)
# mag = np.sqrt(jaaaa.real()**2 + jaaaa.imag()**2)
jaaaa = jaaaa[15000:45000]  # /np.max(jaaaa)

t = np.arange(0, 30000 / samplefreq, 1 / samplefreq)
clock = 0.5 * np.cos(fsy * 2 * np.pi * t + 22 * np.pi / 10) + .5  # vir DBPSK

print(t[0:10])
print(time[0:10])

plt.figure()
plt.plot(np.real(jaaaa), label='filter')
plt.plot(np.real(raw_samples), label='raw_samples real')

# plt.plot(ons_sample_hier_ja, 'go', label='sampling instances')
plt.legend()
plt.yscale('linear')
plt.title('filter')
plt.xlabel('time or samples')
plt.ylabel('not phase')
plt.grid(True)

no_zeros = []
ons_sample_hier_ja = np.zeros([30000], dtype=complex)
flaggie = 0  ##################################################################Die issue is dalk hier
for i in range(len(ons_sample_hier_ja)):
    if (clock[i] >= 0.985) and (ons_sample_hier_ja[i - 1] == 0) and (ons_sample_hier_ja[i - 2] == 0) and (
            ons_sample_hier_ja[i - 3] == 0):
        ons_sample_hier_ja[i] = jaaaa[i]
        if not ((abs(jaaaa[i]) < 0.05) and (abs(jaaaa[i]) > -0.05)):
            flaggie = 1
        if flaggie:
            no_zeros.append(jaaaa[i])

t1 = np.arange(0, len(no_zeros) / samplefreq, 1 / samplefreq)

phase_ons_is_klaar_gesamples = np.unwrap(np.angle(no_zeros)) * -1
# freq_ons_is_klaar_gesamples = np.gradient(phase_ons_is_klaar_gesamples)
mooi_IQ = []

# plt.show()

m = (phase_ons_is_klaar_gesamples[8] - phase_ons_is_klaar_gesamples[3]) / 6
phi = []
phi.append(0.0)

qsam = []  # expected samples
qsam.append(phase_ons_is_klaar_gesamples[0])
# qdiff = []#diff between ultimate expected en value

qdiff = np.zeros([len(phase_ons_is_klaar_gesamples)], dtype=complex)

dtot = 0
diff = []  # cari + phi
qsam = phase_ons_is_klaar_gesamples
pll = np.zeros([len(phase_ons_is_klaar_gesamples)], dtype=complex)
phase_ons_is_klaar_gesamples = np.unwrap(np.angle(no_zeros)) * -1

#################################################################################Hierdie werk tot en met BPSK
diff.append(phase_ons_is_klaar_gesamples[0])
for j in range(len(phase_ons_is_klaar_gesamples) - 1):
    if j < 900:
        diff.append(phase_ons_is_klaar_gesamples[j + 1] - phase_ons_is_klaar_gesamples[j])
        phi.append(diff[j + 1] - m)

        dtot = dtot + phi[j]
        pll[j] = phase_ons_is_klaar_gesamples[j] + dtot

        if j > 5:
            qsam[j] = (qsam[j - 1] + m)
        qdiff[j] = (-phase_ons_is_klaar_gesamples[j] + pll[j])

#################################################################################Hierdie werk tot en met BPSK

sample_phase = np.unwrap(np.angle(no_zeros))
sample_phase = sample_phase[0:850]
pll = sample_phase

sample_phase = np.unwrap(np.angle(no_zeros))
sample_phase = sample_phase[0:850]
new_phase = np.zeros([len(sample_phase)], dtype=complex)

m = (sample_phase[8] - sample_phase[4]) / 5
for k in range(850):
    if k > 8:
        # if ((np.unwrap(np.angle(no_zeros))[k] - np.unwrap(np.angle(no_zeros))[k-1]) > 1*np.pi/8): # or ((pll[k] - pll[k-1]) > 11*np.pi/8)
        pll[k] = pll[k - 1] + m + phi[k]
        new_phase[k] = -pll[k] + sample_phase[k]
        # else:
        # m = (np.unwrap(np.angle(no_zeros))[k]-np.unwrap(np.angle(no_zeros))[k-1])

        # if ((pll[k] - pll[k-1]) < -1*np.pi/8):
        #     pll[k] = pll[k - 1] + m
        # else:
        #     m = (pll[k]-pll[k-1])

plt.figure()
plt.plot(np.real(jaaaa), label='jaaaa real')
# plt.plot(np.real(rawsquare), label='raw ja')
# plt.plot(clock, label='clock')
# plt.plot(ons_sample_hier_ja, 'go', label='sampling instances')
# plt.plot(no_zeros, t, 'bo', label='ons_sample_hier_ja')
plt.legend()
plt.yscale('linear')
plt.title('Bunch of data')
plt.xlabel('time or samples')
plt.ylabel('Amplitude')
plt.grid(True)

plt.figure()
plt.plot(np.real(raw_samples), label='raw real')
# plt.plot(np.real(rawsquare), label='raw ja')
# plt.plot(clock, label='clock')
# plt.plot(ons_sample_hier_ja, 'go', label='sampling instances')
# plt.plot(no_zeros, t, 'bo', label='ons_sample_hier_ja')
plt.legend()
plt.yscale('linear')
plt.title('Bunch of data')
plt.xlabel('time or samples')
plt.ylabel('Amplitude')
plt.grid(True)

plt.figure()
plt.plot(phi, 'ro', label='phi')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('time or samples')
plt.ylabel('phase')
plt.grid(True)

plt.figure()
plt.plot(np.angle(ons_sample_hier_ja), label='samples')
plt.plot(np.angle(raw_samples), label='raw')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('time or samples')
plt.ylabel('phase')
plt.grid(True)

mod_out_hier_sample = []
for i in range(850):
    mod_out_hier_sample.append(abs(no_zeros[i]) * np.exp(
        1j * (phi[i])))  ##############################################################################was phi

out = []

##############################################################DBPSK
for i in range(len(phi)):

    if (phi[i] > 1) or (phi[i] < -2):
        out.append('1')
    else:
        out.append('0')
#################################################################DBPSK

s = ''.join(out)
print(s)

# jas = s[2:-3] #DBPSK
jas = s[20:-1]
print(jas)
bitstr = ''
joke = ""
desi = 0
for i in range(40):
    bitstr = jas[0:8]
    print(bitstr)
    desi = int(bitstr, 2)
    joke = joke + (chr(desi))
    # print(joke)
    jas = jas[8:]

# b = bytes((s[8*i+jas:8*i+8+jas]), 'utf-8')
# print(binascii.b2a_uu(b))  # out[2:-4]))

print(joke)

plt.figure()
plt.plot(np.real(no_zeros), np.imag(no_zeros), 'bo', label='Sample IQ')
plt.plot(np.real(mod_out_hier_sample), np.imag(mod_out_hier_sample), 'ro', label='Sync IQ')
plt.legend()
plt.yscale('linear')
plt.title('Sync IQ diagram')
plt.xlabel('real')
plt.ylabel('imag')
plt.grid(True)

plt.show()

# https://mothereff.in/binary-ascii
# https://www.rapidtables.com/code/text/ascii-table.html


# Veeresh Taranalli, "CommPy: Digital Communication with Python, version 0.3.0. Available at https://github.com/veeresht/CommPy", 2015.
