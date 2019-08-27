import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pylab as py
from rtlsdr import *
import csv
import cmath
import plotly.graph_objects as go
import pandas as pd
from scipy import signal


samples0 = []
samples1 = []
samples2 = []
samples_unwrap = []
samples_unpacked = []
magnetude = []

with open('bursts_weirdness.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    l = 0
    for row in csv_reader :
            #print(row)
            piele = row[2]
            piele = piele.replace("i", "j")
            #complex(''.join(piele.split()))
            samples0.append(complex(piele))
            #samples1.append(cmath.phase(complex(piele))/cmath.pi)
            samples1.append(complex(piele))
            magnetude.append(abs(samples1[l]))
            l += 1
    csv_file.close()

samples_unpacked = np.unwrap(np.angle(samples1))



samples_unwrap = np.zeros([30001],dtype=float)
a=np.zeros([30001],dtype=float)
b=np.zeros([30001],dtype=float)
c=np.zeros([30001],dtype=complex)

predictor = np.zeros([30001],dtype=float)
e = 0
k = 0
d = 0
j = 0


for i in samples_unpacked:
    e = i - predictor[j]
    d = i - samples_unpacked[j-1]
    predictor[j+1] = predictor[j] + e + d
    #print(i)
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

savgol_array = signal.savgol_filter(predictor, 311, 3)

samples_unwrap = np.delete(samples_unwrap, 30000)

print(len(samples_unwrap))
print('samples_unwrap')
print(len(magnetude))
print('magnetude')

for i in range(len(savgol_array)-1):
    samples_unwrap[i] = ( savgol_array[i] + np.pi) % (2 * np.pi ) - np.pi
    a[i] = np.cos(samples_unwrap[i])*magnetude[i]
    b[i] = np.sin(samples_unwrap[i])*magnetude[i]
    c[i] = complex(a[i],b[i])

results_frequency = np.gradient(savgol_array)
samples_frequency = np.gradient(samples1)
#samples_unwrap = complex(np.cos(predictor), np.sin(predictor))

#print(samples_unpacked)
plt.xlabel('samples')
plt.ylabel('Amplificaions')
##sinus tyd
#plot(samples1)
#plot(c)


##Phase
plot(savgol_array)
plot(samples_unpacked)

plt.show()


