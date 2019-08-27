import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pylab as py
from rtlsdr import *
import csv

try:
    sdr = RtlSdr(1)
    sdr.sample_rate = 2.4e6
    sdr.center_freq = 1.3e9
    sdr.gain = 120
    flag = 0
    print('awe1')
except:
    flag = 1
    print('uncool1')
    sdr.close()

if  flag==0:
    print('awe2')
    samples = sdr.read_samples(256*1024*2*2)
    samples1 = samples[200000:300000]
    sdr.close()

    #time_samples = np.real(samples[50000:52000])
    print('awe3')
    '''
    with open('rtslr_out1.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(samples1)

    writeFile.close()
    '''

    # use matplotlib to estimate and plot the PSD
   # plt.psd(samples, NFFT=1024, Fs=sdr.sample_rate / 1e6, Fc=sdr.center_freq / 1e6)
    plt.xlabel('samples')
    plt.ylabel('Amplificaions')

    plot(samples)

    py.savetxt("samples_10.txt", samples, delimiter = ",")

    plt.show()
    #fig, (ax1,ax2) = plt.subplots(2, 1)
    #ax1.psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
    #ax1.set_xlabel('Frequency')
    #ax1.set_ylabel('rel Power [dB]')
    #ax1.set_title('PSD')
    #ax2.plot(time_samples)
    #ax2.set_xlabel('Samples')
    #ax2.set_ylabel('rel Power [dB]')
    #ax2.set_title('time samples')
    #fig.tight_layout()
    #print('awe4')
    #fig.show()
    #print('awe5')