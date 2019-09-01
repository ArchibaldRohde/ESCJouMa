import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pylab as py
from rtlsdr import *
import csv

#########################################################################################################
# This code was copied from clickup

try:
    sdr = RtlSdr(1)
    sdr.sample_rate = 2.4e6
    sdr.center_freq = 1.3e9
    sdr.gain = 30
    flag = 0
    print('awe1')
except:
    flag = 1
    print('uncool1')
    sdr.close()

if flag == 0:
    print('awe2')
    samples = sdr.read_samples(256 * 1024 * 2 * 2)
    sdr.close()

    # time_samples = np.real(samples[50000:52000])
    print('awe3')
    counter = 0
    flag = 0
    j = -1
    for i in range(len(samples)):
        if flag == 0:
            j += 1
            if (samples[i] < 0.05) and (samples[i] > -0.05):
                counter += 1
                if counter > 16500:
                    flag = 1
                    counter = 0

            else:
                counter = 0

    samples0 = samples[j:j + 270000]

    ####Archie
    # samples0 = [w.replace('(', '') for w in samples0]

    ###Awe maar dis complex... so dit werk nie...uhhhmmmm...
    # samples0 = samples0.replace("(", "")
    # samples0 = samples0.replace(")", "")

    samples1 = samples[j:j + 30000]
    samples2 = samples[j + 30000:j + 60000]
    samples3 = samples[j + 60000:j + 90000]
    samples4 = samples[j + 90000:j + 120000]
    samples5 = samples[j + 120000:j + 150000]
    samples6 = samples[j + 150000:j + 180000]
    samples7 = samples[j + 180000:j + 210000]
    samples8 = samples[j + 210000:j + 240000]
    samples9 = samples[j + 240000:j + 270000]
    strsamples1 = []
    strsamples2 = []
    strsamples3 = []
    strsamples4 = []
    strsamples5 = []
    strsamples6 = []
    strsamples7 = []
    strsamples8 = []
    strsamples9 = []

    for l in range(len(samples1)):
        strsamples1.append(str(samples1[l].real) + '+' + str(samples1[l].imag) + 'j')
        strsamples2.append(str(samples2[l].real) + '+' + str(samples2[l].imag) + 'j')
        strsamples3.append(str(samples3[l].real) + '+' + str(samples3[l].imag) + 'j')
        strsamples4.append(str(samples4[l].real) + '+' + str(samples4[l].imag) + 'j')
        strsamples5.append(str(samples5[l].real) + '+' + str(samples5[l].imag) + 'j')
        strsamples6.append(str(samples6[l].real) + '+' + str(samples6[l].imag) + 'j')
        strsamples7.append(str(samples7[l].real) + '+' + str(samples7[l].imag) + 'j')
        strsamples8.append(str(samples8[l].real) + '+' + str(samples8[l].imag) + 'j')
        strsamples9.append(str(samples9[l].real) + '+' + str(samples9[l].imag) + 'j')

    strsamples1 = [w.replace('+-', '-') for w in strsamples1]
    strsamples2 = [w.replace('+-', '-') for w in strsamples2]
    strsamples3 = [w.replace('+-', '-') for w in strsamples3]
    strsamples4 = [w.replace('+-', '-') for w in strsamples4]
    strsamples5 = [w.replace('+-', '-') for w in strsamples5]
    strsamples6 = [w.replace('+-', '-') for w in strsamples6]
    strsamples7 = [w.replace('+-', '-') for w in strsamples7]
    strsamples8 = [w.replace('+-', '-') for w in strsamples8]
    strsamples9 = [w.replace('+-', '-') for w in strsamples9]

    '''
    strsamples1 = [w.replace('', '\n') for w in strsamples1]
    strsamples2 = [w.replace('\n\n', '\n') for w in strsamples2]
    strsamples3 = [w.replace('\n\n', '\n') for w in strsamples3]
    strsamples4 = [w.replace('\n\n', '\n') for w in strsamples4]
    strsamples5 = [w.replace('\n\n', '\n') for w in strsamples5]
    strsamples6 = [w.replace('\n\n', '\n') for w in strsamples6]
    strsamples7 = [w.replace('\n\n', '\n') for w in strsamples7]
    strsamples8 = [w.replace('\n\n', '\n') for w in strsamples8]
    strsamples9 = [w.replace('\n\n', '\n') for w in strsamples9]
    '''

    # strsamples1 = strsamples1.replace("+-","-")
    print(strsamples1)
    # print('samples2: {:.2f}'.format(samples2))
    # list(map('{:.2f}%'.format, samples2))
    '''list_weird = [[]]
    for k in range(29999):
        list_weird[k] = [samples1[k], samples2[k], samples3[k], samples4[k], samples5[k], samples6[k], samples7[k], samples8[k], samples9[k]]
    '''

    list_weird = []
    # print(list_weird)
    # list_weird = list(filter(None, list_weird))

    # list_weird1 = zip(*list_weird)


    # use matplotlib to estimate and plot the PSD
   # plt.psd(samples, NFFT=1024, Fs=sdr.sample_rate / 1e6, Fc=sdr.center_freq / 1e6)
plt.xlabel('samples')
plt.ylabel('Amplificaions')
#plot(samples)
plt.figure(0)
plot(samples0)
plt.figure(1)
plot(samples1)
plt.figure(2)
plot(samples2)
plt.figure(3)
plot(samples3)
plt.figure(4)
plot(samples4)
plt.figure(5)
plot(samples5)
plt.figure(6)
plot(samples6)
plt.figure(7)
plot(samples7)
plt.figure(8)
plot(samples8)
plt.figure(9)
plot(samples9)

    #########################################################################################################

#py.savetxt("samples_10.txt", samples, delimiter = ",")

plt.show()



for h in range(30000):
    list_weird.append(
        [strsamples1[h], strsamples2[h], strsamples3[h], strsamples4[h], strsamples5[h], strsamples6[h],
         strsamples7[h], strsamples8[h], strsamples9[h]])
    with open('OOK.txt', 'a') as the_file:
        the_file.write(strsamples1[h] + ',' + '\n')
    the_file.close()
    with open('ASK.txt', 'a') as the_file:
        the_file.write(strsamples2[h] + ',' + '\n')
    the_file.close()
    with open('BPSK.txt', 'a') as the_file:
        the_file.write(strsamples3[h] + ',' + '\n')
    the_file.close()
    with open('DBPSK.txt', 'a') as the_file:
        the_file.write(strsamples4[h] + ',' + '\n')
    the_file.close()
    with open('DQPSK.txt', 'a') as the_file:
        the_file.write(strsamples5[h] + ',' + '\n')
    the_file.close()
    with open('D8PSK.txt', 'a') as the_file:
        the_file.write(strsamples6[h] + ',' + '\n')
    the_file.close()
    with open('MSK.txt', 'a') as the_file:
        the_file.write(strsamples7[h] + ',' + '\n')
    the_file.close()
    with open('QAM.txt', 'a') as the_file:
        the_file.write(strsamples8[h] + ',' + '\n')
    the_file.close()
    with open('16QAM.txt', 'a') as the_file:
        the_file.write(strsamples9[h] + ',' + '\n')
    the_file.close()