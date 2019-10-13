import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.pyplot import *
import pylab as py
# from rtlsdr import *
import csv
import math
# import plotly.graph_objects as go
# import pandas as pd
import scipy as sc  #
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from bitstring import BitArray

#################Read From Textfile##################

raw_samples = []
with open("Convolution.txt") as fileobject:
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
clock = 0.5*np.cos(fsy*2*np.pi*t + 30*np.pi/20) + .5 #vir DBPSK


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
    if j < len(phase_ons_is_klaar_gesamples):
        diff.append(phase_ons_is_klaar_gesamples[j + 1] - phase_ons_is_klaar_gesamples[j])
        phi.append(diff[j + 1] - m)

        dtot = dtot + phi[j]
        pll[j] = phase_ons_is_klaar_gesamples[j] + dtot

        if j > 5:
            qsam[j] = (qsam[j - 1] + m)
        qdiff[j] = (-phase_ons_is_klaar_gesamples[j] + pll[j])

#################################################################################Hierdie werk tot en met BPSK

sample_phase = np.unwrap(np.angle(no_zeros))
#sample_phase = sample_phase[0:850]
pll = sample_phase

sample_phase = np.unwrap(np.angle(no_zeros))
#sample_phase = sample_phase[0:len]
new_phase = np.zeros([len(sample_phase)], dtype=complex)

m = (sample_phase[8] - sample_phase[4]) / 5
for k in range(len(phi)):
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
for i in range(len(phi)):
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
print(len(s))
print("bier")


######################################################################################Wierd FSM stuff


n=0
stront = ""
bitstr = ''
s = s[10:] #speel met dit as niks werk nie
print(s)

Data = ["00000", "10110", "11101", "01011", "11010", "01100", "00111", "10001"]

state000 = True
state001 = False
state010 = False
state011 = False
state100 = False
state101 = False
state110 = False
state111 = False

for i in range(math.floor((len(s)/6))):
    code = s[0:5]
    b = BitArray(bin=code)
    s = s[5:]
    nextcode = s[0:5]
    nextb = BitArray(bin=code)

    if state000:
        a01 = np.count_nonzero(b ^ BitArray(bin=Data[0]))
        a11 = np.count_nonzero(b ^ BitArray(bin=Data[1]))
        a21 = np.count_nonzero(b ^ BitArray(bin=Data[2]))
        a31 = np.count_nonzero(b ^ BitArray(bin=Data[3]))

        ax0 = coolcool.index(min(coolcool))

        a0 = a01 + ax0
        a1 = a11 + ax1
        a2 = a21 + ax2
        a3 = a31 + ax3

        coolcool = [a0, a1, a2, a3]
        idex = coolcool.index(min(coolcool))
        del(coolcool[idex])
        idex2 = coolcool.index(min(coolcool))

        # FSM next state
        if (idex == 0):
            state000 = True
            state001 = False
            state010 = False
            state011 = False
            state100 = False
            state101 = False
            state110 = False
            state111 = False
            stront = stront + "00"

        elif (idex == 1):
            state000 = False
            state001 = True
            state010 = False
            state011 = False
            state100 = False
            state101 = False
            state110 = False
            state111 = False
            stront = stront + "01"
        elif (idex == 2):
            state000 = False
            state001 = False
            state010 = True
            state011 = False
            state100 = False
            state101 = False
            state110 = False
            state111 = False
            stront = stront + "10"
        elif (idex == 3):
            state000 = False
            state001 = False
            state010 = False
            state011 = True
            state100 = False
            state101 = False
            state110 = False
            state111 = False
            stront = stront + "11"
        else:
            print("FOOOOOOOK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#Data = ["00000", "10110", "11101", "01011", "11010", "01100", "00111", "10001"]



    elif state001:
        a0 = np.count_nonzero(b ^ BitArray(bin=Data[4]))
        a1 = np.count_nonzero(b ^ BitArray(bin=Data[5]))
        a2 = np.count_nonzero(b ^ BitArray(bin=Data[6]))
        a3 = np.count_nonzero(b ^ BitArray(bin=Data[7]))
        coolcool = [a0, a1, a2, a3]
        idex = coolcool.index(min(coolcool))
        # FSM next state
        if (idex == 0):
            state000 = False
            state001 = False
            state010 = False
            state011 = False
            state100 = True
            state101 = False
            state110 = False
            state111 = False
            stront = stront + "00"

        elif (idex == 1):
            state000 = False
            state001 = False
            state010 = False
            state011 = False
            state100 = False
            state101 = True
            state110 = False
            state111 = False
            stront = stront + "01"
        elif (idex == 2):
            state000 = False
            state001 = False
            state010 = False
            state011 = False
            state100 = False
            state101 = False
            state110 = True
            state111 = False
            stront = stront + "10"
        elif (idex == 3):
            state000 = False
            state001 = False
            state010 = False
            state011 = False
            state100 = False
            state101 = False
            state110 = False
            state111 = True
            stront = stront + "11"
        else:
            print("FOOOOOOOK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#Data = ["00000", "10110", "11101", "01011", "11010", "01100", "00111", "10001"]



    else:
        print("!Dubbel gefok!")

#Data = ["00000", "10110", "11101", "01011", "11010", "01100", "00111", "10001"]

    print(stront)




######################################################################################Wierd FSM stuff

#jas = s[2:-3] #DBPSK
jas = s
bitstr = ''
joke = ''
desi = 0
for i in range(math.floor((len(jas)/8))):
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


#Veeresh Taranalli, "CommPy: Digital Communication with Python, version 0.3.0. Available at https://github.com/veeresht/CommPy", 2015.