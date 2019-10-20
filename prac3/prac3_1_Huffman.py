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

#################Read From Textfile##################

raw_samples = []
with open("Huffman.txt") as fileobject:
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
fsymrate = 1.3e9 + fsy

t = np.arange(0, 30000 / samplefreq, 1 / samplefreq)
clock = 0.5 * np.cos(fsy * 2 * np.pi * t + 37 * np.pi / 20) + .5  # vir DBPSK

no_zeros = []
ons_sample_hier_ja = np.zeros([30000], dtype=complex)
flaggie = 0  ##################################################################Die issue is dalk hier
for i in range(len(ons_sample_hier_ja)):
    if (clock[i] >= 0.985) and (ons_sample_hier_ja[i - 1] == 0) and (ons_sample_hier_ja[i - 2] == 0) and (
            ons_sample_hier_ja[i - 3] == 0):
        ons_sample_hier_ja[i] = raw_samples[i]
        if not ((abs(raw_samples[i]) < 0.05) and (abs(raw_samples[i]) > -0.05)):
            flaggie = 1
        if flaggie:
            no_zeros.append(raw_samples[i])

t1 = np.arange(0, len(no_zeros) / samplefreq, 1 / samplefreq)

phase_ons_is_klaar_gesamples = np.unwrap(np.angle(no_zeros)) * -1
# freq_ons_is_klaar_gesamples = np.gradient(phase_ons_is_klaar_gesamples)
mooi_IQ = []

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
plt.plot(np.real(raw_samples), label='raw_samples real')
plt.plot(np.imag(raw_samples), label='raw_samples imag')
plt.plot(clock, label='clock')
plt.plot(ons_sample_hier_ja, 'go', label='sampling instances')
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
plt.plot(np.unwrap(np.angle(no_zeros)), label='samples')
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

    if (phi[i] > 2) or (phi[i] < -2):
        out.append('1')
    else:
        out.append('0')
#################################################################DBPSK

s = ''.join(out)
print(s)

######################################################################################Archie se dodgy hufflepuff

n = 0
stront = ""
s = s[1:-1]  # Huffman
while n < (len(s) - 12):

    if s[n:n + 3] == "000":
        stront = stront + " "
        n = n + 3
    elif s[n:n + 4] == "0011":
        stront = stront + "T"
        n = n + 4
    elif s[n:n + 6] == "001001":
        stront = stront + "C"
        n = n + 6
    elif s[n:n + 8] == "00100010":
        stront = stront + "$"
        n = n + 8
    elif s[n:n + 7] == "0010000":
        stront = stront + "."
        n = n + 7
    elif s[n:n + 8] == "00100011":
        stront = stront + "^"
        n = n + 8
    elif s[n:n + 6] == "001010":
        stront = stront + "U"
        n = n + 6
    elif s[n:n + 7] == "0010111":
        stront = stront + "V"
        n = n + 7
    elif s[n:n + 8] == "00101101":
        stront = stront + "`"
        n = n + 8
    elif s[n:n + 8] == "00101100":
        stront = stront + ","
        n = n + 8
    elif s[n:n + 6] == "010000":
        stront = stront + "M"
        n = n + 6
    elif s[n:n + 6] == "010001":
        stront = stront + "W"
        n = n + 6
    elif s[n:n + 5] == "01001":
        stront = stront + "D"
        n = n + 5
    elif s[n:n + 6] == "010100":
        stront = stront + "F"
        n = n + 6
    elif s[n:n + 6] == "010101":
        stront = stront + "G"
        n = n + 6
    elif s[n:n + 5] == "01011":
        stront = stront + "L"
        n = n + 5
    elif s[n:n + 4] == "0110":
        stront = stront + "A"
        n = n + 4
    elif s[n:n + 4] == "0111":
        stront = stront + "O"
        n = n + 4
    elif s[n:n + 6] == "100000":
        stront = stront + "Y"
        n = n + 6
    elif s[n:n + 6] == "100001":
        stront = stront + "P"
        n = n + 6
    elif s[n:n + 9] == "100010000":
        stront = stront + "``"
        n = n + 9
    elif s[n:n + 11] == "10001000100":
        stront = stront + "?"
        n = n + 11
    elif s[n:n + 11] == "10001000101":
        stront = stront + "Z"
        n = n + 11
    elif s[n:n + 10] == "1000100011":
        stront = stront + "Q"
        n = n + 10
    elif s[n:n + 9] == "100010010":
        stront = stront + "J"
        n = n + 9
    elif s[n:n + 9] == "100010011":
        stront = stront + "X"
        n = n + 9
    elif s[n:n + 7] == "1000101":
        stront = stront + "K"
        n = n + 7
    elif s[n:n + 6] == "100011":
        stront = stront + "B"
        n = n + 6
    elif s[n:n + 4] == "1001":
        stront = stront + "I"
        n = n + 4
    elif s[n:n + 4] == "1010":
        stront = stront + "N"
        n = n + 4
    elif s[n:n + 4] == "1011":
        stront = stront + "S"
        n = n + 4
    elif s[n:n + 3] == "110":
        stront = stront + "E"
        n = n + 3
    elif s[n:n + 4] == "1110":
        stront = stront + "H"
        n = n + 4
    elif s[n:n + 4] == "1111":
        stront = stront + "R"
        n = n + 4
    else:
        n = n + 1
        print("Fok")

print(stront)
######################################################################################Archie se dodgy hufflepuff


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
