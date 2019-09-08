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
#fsy = 93728 #vir ASK
fsy = 102732 #vir BPSK
#fsy = 94929 #vir D8PSK
#fsy = 102731 #vir DBPSK
#fsy = 96144 #vir DQPSK
#fsy = 96144 #vir MSK ###########################Hierdie een is weird
#fsy = 99992 #vir OOK
#fsy = 104160 #vir QAM



samplefreq = 2.4e6
fsymrate = 1.3e9 + fsy

t = np.arange(0, 30000/samplefreq, 1/samplefreq)
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi+1.5) + .5  #Vir 16QAM
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi-0.6) + .5 #vir ASK
clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi*2/20) + .5 #vir BPSK
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi+0.2) + .5 #vir D8PSK
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi+1.5) + .5 #vir DBPSK
#clock = 0.5*np.cos(fsy*2*np.pi*t+0.5) + .5 #vir DQPSK
#clock = 0.5*np.cos(fsy*2*np.pi*t+0.5) + .5 #vir MSK ################################Hierdie een is weird
#clock = 0.5*np.cos(fsy*2*np.pi*t+0.4) + .5 #vir OOK
#clock = 0.5*np.cos(fsy*2*np.pi*t+np.pi-0.2) + .5 #vir QAM



no_zeros = []
ons_sample_hier_ja = np.zeros([30000], dtype=complex)
flaggie = 0
for i in range(len(ons_sample_hier_ja)):
    if (clock[i] >= 0.985) and (ons_sample_hier_ja[i-1] == 0)and (ons_sample_hier_ja[i-2] == 0)and (ons_sample_hier_ja[i-3] == 0):
        ons_sample_hier_ja[i] = raw_samples[i]
        if not((abs(raw_samples[i]) < 0.05) and (abs(raw_samples[i]) > -0.05)):
            flaggie = 1
        if flaggie:
            no_zeros.append(raw_samples[i])




phase_ons_is_klaar_gesamples = np.unwrap(np.angle(no_zeros))*-1
freq_ons_is_klaar_gesamples = np.gradient(phase_ons_is_klaar_gesamples)
mooi_IQ = []


m = (phase_ons_is_klaar_gesamples[8] - phase_ons_is_klaar_gesamples[3])/5
phi = []
phi.append(0.0)
diff = []#cari + phi

diff.append(phase_ons_is_klaar_gesamples[0])
for j in range(len(phase_ons_is_klaar_gesamples)-1):
    if j < 900:
        if (m - abs(phase_ons_is_klaar_gesamples[j+1] - phase_ons_is_klaar_gesamples[j]) < 0.0005):
            m = abs(phase_ons_is_klaar_gesamples[j+1] - phase_ons_is_klaar_gesamples[j])
        diff.append(phase_ons_is_klaar_gesamples[j+1] - phase_ons_is_klaar_gesamples[j])
        phi.append(diff[j+1] - m)

#phi[0] = 0
#phi[1] = 0
#phi[2] = 0
#phi[3] = 0
#phi[4] = 0
#phi[5] = 0#################################################################################################################Die eerste klompie samples is bogus
#phi[6] = 0
#phi[7] = 0
#phi[8] = 0
#phi[9] = 0
#phi[10] = 0
#phi[11] = 0

out = []

for i in phi:
    if ((i > ((-3)*(np.pi/2) ) )and ((i < (-np.pi)))):
            out.append('1')
    else:
        out.append('0')
print(out)


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

mod_out_hier_sample = []
for i in range(len(no_zeros)):
    if i < 900:
        mod_out_hier_sample.append(abs(no_zeros[i])*np.exp(1j*phi[i]))

freq_ons_is_klaar_gesamples = freq_ons_is_klaar_gesamples +0.75 #BPSK
for i in range(len(no_zeros)):
    if (freq_ons_is_klaar_gesamples[i] >= np.pi*0.9) and (freq_ons_is_klaar_gesamples[i] <= np.pi*1.1):
        break#hy het gejump


plt.figure()
plt.plot(raw_samples, label='raw_samples')
plt.plot(clock, label='clock')
plt.plot(ons_sample_hier_ja, 'go', label='ons_sample_hier_ja')
#plt.plot(no_zeros, t, 'bo', label='ons_sample_hier_ja')
plt.legend()
plt.yscale('linear')
plt.title('phase_diff')
plt.xlabel('time')
plt.ylabel('Modulation')
plt.grid(True)


plt.figure()
plt.plot(np.real(no_zeros), np.imag(no_zeros), 'bo', label='modulation_out IQ')
plt.plot(np.real(mod_out_hier_sample), np.imag(mod_out_hier_sample), 'ro', label='modulation_out IQ')

plt.legend()
plt.yscale('linear')
plt.title('phase_diff')
plt.xlabel('real')
plt.ylabel('imag')
plt.grid(True)


plt.figure()
plt.plot(np.real(mod_out_hier_sample), np.imag(mod_out_hier_sample), 'ro', label='modulation_out IQ')
plt.legend()
plt.yscale('linear')
plt.title('phase_diff')
plt.xlabel('real')
plt.ylabel('imag')
plt.grid(True)


#PLL phase
plt.figure()
#plt.subplot(221)
plt.plot(phi, label='phase_ons_is_klaar_gesamples')
#plt.plot(pll_out, label='PLL')
plt.legend()
plt.yscale('linear')
plt.title('PLL')
plt.xlabel('samples')
plt.ylabel('Phase')
plt.grid(True)
plt.figure()
'''

#plt.subplot(221)
plt.plot(modulation_out, label='modulation_out')
plt.legend()
plt.yscale('linear')
plt.title('phase_diff')
plt.xlabel('samples')
plt.ylabel('Phase')
plt.grid(True)
plt.figure()
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

f_s = 2.4e6 ####################################################################################Archie copied this and put it higher up yass

Y = np.abs(raw_samples[1000:29000])
X = np.abs(sc.fft(Y,300000))
fmag1 = 10 * np.log(X/max(X))
fmag2 = 20 * np.log(X/max(X))

freqs = sc.fftpack.fftfreq(len(fmag1)) *f_s#/len(fmag)##############################################
'''
plt.figure()
#plt.subplot(221)
plt.plot(modulation_out,  label=' samples of sigal')
#plt.plot(symrateout, 'go', label=' symrate out samples')
plt.legend()
#plt.yscale('samplesize')
plt.title('symrateoversamples')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid(True)
'''


plt.figure()

fig, ax = plt.subplots()
#ax.yscale('linear')
#plt.title('Fourier transform')
#ax.ylabel('frequency[Hz]')
#ax.ylabel('Amplitude[dB]')
#ax.legend()
ax.plot(freqs, fmag2, label='fourier transform')
#plt.plot(freqs, fmag1, label='fourier transform')



plt.show()