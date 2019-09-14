import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
from matplotlib.pyplot import *
import pylab as py
#from rtlsdr import *
import csv
import cmath
#import plotly.graph_objects as go
#import pandas as pd
import scipy as sc#
from scipy import signal
from scipy.signal import butter, lfilter, freqz
#from scipy import fftpack
import binascii

#for g in range(11):
#leerman = np.zeros([30000], dtype=str)
#raw_samples = np.zeros([30000], dtype=complex)

#################Read From Textfile##################

raw_samples = []
with open("DQPSK.txt") as fileobject:
    l = 0
    for line in fileobject:
        reading = line[0: line.find(",")]
        raw_samples.append(complex(reading))
        l=+1
fileobject.close()
#print(raw_samples)

#################Read From Textfile##################


'''
phase_raw_samples = np.unwrap(np.angle(raw_samples))*-1
#phase_raw_samples = sc.signal.savgol_filter(phase_raw_samples1, 11, 3)
pll_out = np.zeros([30000], dtype=complex)
new_phase_raw_samples = np.zeros([30000], dtype=complex)
#filter noise out
#try wrapping phase and filtering then unwrap..
counter = 0
m = 0
plekhouer = 0
flaggie = 0
for i in range(len(phase_raw_samples)-1):
    #print(phase_raw_samples[i])
    if flaggie == 0:
        if (phase_raw_samples[i] > 150):
            if (counter < 50):
                #print("####################")
               # print(phase_raw_samples[i])
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
'''

'''
######################################################################################################################Archie pasted this from down below
f_s = 2.4e6
Y = np.abs(raw_samples[1000:29000])
X = np.abs(sc.fft(Y,300000))
fmag1 = 10 * np.log(X/max(X))
fmag2 = 20 * np.log(X/max(X))
freqs = sc.fftpack.fftfreq(len(fmag1)) *f_s#/len(fmag)
######################################################################################################################
#fsy = (np.argmax(freqs[1000:400000])+1000)
#print(fsy)
'''
#fsy = 98684 #vir 16QAM
#fsy = 102728#93728 #vir ASK
#fsy = 97396#102732 #vir BPSK
#fsy = 94929 #vir D8PSK
#fsy = 93744#102731 #vir DBPSK
fsy = 101344#96144 #vir DQPSK
#fsy = 96144 #vir MSK ###########################Hierdie een is weird
#fsy = 94927#99992 #vir OOK
#fsy = 104160 #vir QAM



samplefreq = 2.4e6
fsymrate = 1.3e9 + fsy

t = np.arange(0, 30000/samplefreq, 1/samplefreq)
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi+1.5) + .5  #Vir 16QAM
#clock = 0.5*np.cos(fsy*2*np.pi*t + 19*np.pi/40) + .5 #vir ASK    np.pi-0.6
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi*119/80) + .5 #vir BPSK
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi+0.2) + .5 #vir D8PSK
#clock = 0.5*np.cos(fsy*2*np.pi*t + 54*np.pi/40) + .5 #vir DBPSK 54*np.pi/40
clock = 0.5*np.cos(fsy*2*np.pi*t + 132*np.pi/20) + .5 #vir DQPSK
#clock = 0.5*np.cos(fsy*2*np.pi*t+0.5) + .5 #vir MSK ################################Hierdie een is weird
#clock = 0.5*np.cos(fsy*2*np.pi*t + 8*np.pi/5) + .5 #vir OOK 0.4
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi-0.2) + .5 #vir QAM



no_zeros = []
ons_sample_hier_ja = np.zeros([30000], dtype=complex)
flaggie = 0##################################################################Die issue is dalk hier
for i in range(len(ons_sample_hier_ja)):
    if (clock[i] >= 0.985) and (ons_sample_hier_ja[i-1] == 0)and (ons_sample_hier_ja[i-2] == 0)and (ons_sample_hier_ja[i-3] == 0):
        ons_sample_hier_ja[i] = raw_samples[i]
        if not((abs(raw_samples[i]) < 0.05) and (abs(raw_samples[i]) > -0.05)):
            flaggie = 1
        if flaggie:
            no_zeros.append(raw_samples[i])



t1 = np.arange(0, len(no_zeros)/samplefreq, 1/samplefreq)

phase_ons_is_klaar_gesamples = np.unwrap(np.angle(no_zeros))*-1
#freq_ons_is_klaar_gesamples = np.gradient(phase_ons_is_klaar_gesamples)
mooi_IQ = []



m = (phase_ons_is_klaar_gesamples[8] - phase_ons_is_klaar_gesamples[3])/6
phi = []
phi.append(0.0)


qsam = []#expected samples
qsam.append(phase_ons_is_klaar_gesamples[0])
#qdiff = []#diff between ultimate expected en value

qdiff = np.zeros([len(phase_ons_is_klaar_gesamples)], dtype=complex)

dtot = 0
diff = []#cari + phi
qsam = phase_ons_is_klaar_gesamples
pll = np.zeros([len(phase_ons_is_klaar_gesamples)], dtype=complex)
phase_ons_is_klaar_gesamples = np.unwrap(np.angle(no_zeros))*-1

#################################################################################Hierdie werk tot en met BPSK
diff.append(phase_ons_is_klaar_gesamples[0])
for j in range(len(phase_ons_is_klaar_gesamples)-1):
    if j < 900:
        if (m - (abs(phase_ons_is_klaar_gesamples[j+1] - phase_ons_is_klaar_gesamples[j])) < 0.0005):
            #m = abs(phase_ons_is_klaar_gesamples[j+1] - phase_ons_is_klaar_gesamples[j])
            print(m)
        diff.append(phase_ons_is_klaar_gesamples[j+1] - phase_ons_is_klaar_gesamples[j])
        phi.append(diff[j+1] - m)

        dtot = dtot + phi[j]
        pll[j] = phase_ons_is_klaar_gesamples[j] + dtot


        if j > 5:
            qsam[j] = (qsam[j-1]+m)
        qdiff[j] = (-phase_ons_is_klaar_gesamples[j] + pll[j])

#################################################################################Hierdie werk tot en met BPSK

sample_phase = np.unwrap(np.angle(no_zeros))
sample_phase = sample_phase[0:850]
pll = sample_phase

sample_phase = np.unwrap(np.angle(no_zeros))
sample_phase = sample_phase[0:850]
new_phase = np.zeros([len(sample_phase)], dtype=complex)

m = (sample_phase[8] - sample_phase[4])/5
for k in range(850):
    if k >8:
        #if ((np.unwrap(np.angle(no_zeros))[k] - np.unwrap(np.angle(no_zeros))[k-1]) > 1*np.pi/8): # or ((pll[k] - pll[k-1]) > 11*np.pi/8)
        pll[k] = pll[k-1]+ m + phi[k]
        new_phase[k] = -pll[k] + sample_phase[k]
        #else:
            #m = (np.unwrap(np.angle(no_zeros))[k]-np.unwrap(np.angle(no_zeros))[k-1])


        # if ((pll[k] - pll[k-1]) < -1*np.pi/8):
        #     pll[k] = pll[k - 1] + m
        # else:
        #     m = (pll[k]-pll[k-1])


plt.figure()
plt.plot(phi,  label='phi')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('time or samples')
plt.ylabel('phase')
plt.grid(True)
plt.figure()
plt.plot(np.unwrap(np.angle(no_zeros)),  label='samples')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('time or samples')
plt.ylabel('phase')
plt.grid(True)

mod_out_hier_sample = []
for i in range(850):
    mod_out_hier_sample.append(abs(no_zeros[i])*np.exp(1j*(phi[i])))##############################################################################was phi

# freq_ons_is_klaar_gesamples = freq_ons_is_klaar_gesamples +0.75 #BPSK
# for i in range(len(no_zeros)):
#     if (freq_ons_is_klaar_gesamples[i] >= np.pi*0.9) and (freq_ons_is_klaar_gesamples[i] <= np.pi*1.1):
#         break#hy het gejump


out = []
#b = 0


# ##############################################################DBPSK
# for i in range(len(phi)):
#
#     if (phi[i] > 2) or (phi[i] < -2):
#         out.append('1')
#     else:
#         out.append('0')
# #################################################################BPSK

# ################################################################OOK
# a = np.real(mod_out_hier_sample)
# b = np.imag(mod_out_hier_sample)
# c = np.sqrt(b**2 + a**2)
# for i in range(len(mod_out_hier_sample)):
#      if (c[i] > -0.15) and (c[i] < 0.15):
#          out.append('0')
#      else:
#          out.append('1')
# ################################################################OOK

# ################################################################DQPSK

angel = 0
# for j in range(len(phi)):
#     phi[j] = -1 * phi[j]

for k in range(len(phi)):
    print(angel)
    if angel == 0:
        if (phi[k] > -np.pi/4) and (phi[k] < np.pi/4):
            angel = 0
        elif (phi[k] < 3*np.pi/4) and (phi[k] > np.pi/4):#pi/2
            angel = 3*np.pi/2
        elif (phi[k] > 3*np.pi/4) and (phi[k] < 5*np.pi/4):#pi
            angel = np.pi
        elif (phi[k] < -1*np.pi/4):# and (phi[k] > np.pi/4):#3pi/2
            angel = np.pi/2
    elif angel == np.pi/2:
        if (phi[k] > -np.pi/4) and (phi[k] < np.pi/4):
            angel =  np.pi/2
        elif (phi[k] < 3*np.pi/4) and (phi[k] > np.pi/4):#pi/2
            angel = 0
        elif (phi[k] > 3*np.pi/4) and (phi[k] < 5*np.pi/4):#pi
            angel = 3*np.pi/2
        elif (phi[k] < -1*np.pi/4):# and (phi[k] > np.pi/4):#3pi/2
            angel = np.pi
    elif angel == np.pi:
        if (phi[k] > -np.pi/4) and (phi[k] < np.pi/4):
            angel = np.pi
        elif (phi[k] < 3*np.pi/4) and (phi[k] > np.pi/4):#pi/2
            angel = np.pi/2
        elif (phi[k] > 3*np.pi/4) and (phi[k] < 5*np.pi/4):#pi
            angel = 0
        elif (phi[k] < -1*np.pi/4):# and (phi[k] > np.pi/4):#3pi/2
            angel = 3*np.pi/2
    elif angel == 3*np.pi/2:
        if (phi[k] > -np.pi/4) and (phi[k] < np.pi/4):
            angel = 3*np.pi/2
        elif (phi[k] < 3*np.pi/4) and (phi[k] > np.pi/4):#pi/2
            angel = np.pi
        elif (phi[k] > 3*np.pi/4) and (phi[k] < 5*np.pi/4):#pi
            angel = np.pi/2
        elif (phi[k] < -1*np.pi/4):# and (phi[k] > np.pi/4):#3pi/2
            angel = 0


    #angel = angel*-1
    if (angel == 0):
        out.append('00')
    elif (angel == np.pi/2):
        out.append('01')
    elif (angel == np.pi):
        out.append('11')
    elif (angel == 3*np.pi/2):
        out.append('10')
    else:
        out.append('22')
        print('FOK')

# if (angel == 0):
    #     out.append('00')
    # elif (angel == np.pi/2):
    #     out.append('01')
    # elif (angel == np.pi):
    #     out.append('11')
    # else:
    #     out.append('10')

# ################################################################DQPSK


'''
wrappedqdiff = np.angle(np.exp(1j*qdiff))
for i in range(len(wrappedqdiff)):
    if (wrappedqdiff[i] > 2):
        out.append('1')
    else:
        out.append('0')
###################################################################

'''
s = ''.join(out)
print(s)
print('#######################')
jas = s[2:-3]
print(jas)
bitstr = ''
joke = ''
desi = 0
for i in range(80):
    bitstr = jas[0:8]
    print(bitstr)
    desi = int(bitstr,2)
    joke = joke + (chr(desi))
    #print(joke)
    jas = jas[8:]

#b = bytes((s[8*i+jas:8*i+8+jas]), 'utf-8')
#print(binascii.b2a_uu(b))  # out[2:-4]))

print(joke)
'''
############################################################################################output string for d8psk
out = []
for i in phi:
    if ((i > (-0.5534-np.pi) ) and (i < 0.2335 -np.pi)):
            out.append('000')
    elif ((i > (0.2335-np.pi) ) and (i < (1.05-np.pi) )):
            out.append('001')
    elif ((i > (1.05-np.pi) ) and (i < (1.77-np.pi) )):
            out.append('011')
    elif ((i > 1.77 -np.pi) and (i < 2.61 -np.pi)):
            out.append('010')
    elif ((i > (-np.pi-np.pi)) and (i < (-2.967-np.pi))) or ((i > 2.61-np.pi) and (i < np.pi-np.pi)):
        out.append('110')
    elif ((i > (-2.967-np.pi) ) and ((i < -2.114-np.pi))):
            out.append('111')
    elif ((i > (-2.114-np.pi) ) and (i < (-1.83-np.pi) )):
            out.append('101')
    elif ((i > (-1.83-np.pi)) and (i < (-0.5534-np.pi))):
        out.append('100')
    else:
        out.append('222')
print(out)
########################################################################################
'''

#diff = diff + np.pi/4



plt.figure()
plt.plot(np.real(raw_samples), label='raw_samples real')
plt.plot(np.imag(raw_samples), label='raw_samples imag')
plt.plot(clock, label='clock')
plt.plot(ons_sample_hier_ja, 'go', label='sampling instances')
#plt.plot(no_zeros, t, 'bo', label='ons_sample_hier_ja')
plt.legend()
plt.yscale('linear')
plt.title('Bunch of data')
plt.xlabel('time or samples')
plt.ylabel('Amplitude')
plt.grid(True)

plt.figure()
plt.plot(no_zeros, 'go')
plt.plot(no_zeros)
#plt.plot(no_zeros, t, 'bo', label='ons_sample_hier_ja')
plt.legend()
plt.yscale('linear')
plt.title('Sampling instances')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.figure()
plt.plot(np.real(no_zeros), np.imag(no_zeros), 'bo', label='Sample IQ')
plt.plot(np.real(mod_out_hier_sample), np.imag(mod_out_hier_sample), 'ro', label='Sync IQ')
plt.legend()
plt.yscale('linear')
plt.title('Sync IQ diagram')
plt.xlabel('real')
plt.ylabel('imag')
plt.grid(True)


plt.figure()
plt.plot(np.real(mod_out_hier_sample), np.imag(mod_out_hier_sample), 'ro', label='Sync IQ')
plt.legend()
plt.yscale('linear')
plt.title('Sync IQ diagram')
plt.xlabel('real')
plt.ylabel('imag')
plt.grid(True)

'''
#PLL phase
plt.figure()
#plt.subplot(221)
plt.plot(phase_ons_is_klaar_gesamples, label='phase_ons_is_klaar_gesamples')
plt.plot(qsam, label='qsam')
plt.plot(qdiff, label='qdiff')
#plt.plot(pll_out, label='PLL')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('samples')
plt.ylabel('Phase')
plt.grid(True)
'''
f_s = 2.4e6 ####################################################################################Archie copied this and put it higher up yass

Y = np.abs(raw_samples[1000:29000])
X = np.abs(sc.fft(Y,300000))
fmag1 = 10 * np.log(X/max(X))
fmag2 = 20 * np.log(X/max(X))

freqs = sc.fftpack.fftfreq(len(fmag1)) *f_s#/len(fmag)##############################################
#freqs1 = np.fft.fftshift(freqs)
fig, ax = plt.subplots()
#ax.yscale('linear')
plt.title('Fourier transform')
plt.xlabel('frequency[Hz]')
plt.ylabel('Amplitude[dB]')
plt.legend()
plt.plot(freqs, fmag2, label='fourier transform')
#plt.plot(freqs, fmag1, label='fourier transform')
plt.show()


#https://mothereff.in/binary-ascii
#https://www.rapidtables.com/code/text/ascii-table.html