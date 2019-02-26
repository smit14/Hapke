#import libraries

from __future__ import division
import os
from subprocess import call

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,tan
import math

# Dataset

#file names of dataset
    
#461 reflectance per sample and total 19 samples 
Data = np.zeros([461,19])    

#data files --> 9 mare + 10 highland
files = pd.read_csv('Dataset/abundance_data.csv',usecols=[0],header=None)
files = files.values


'''
sample order:

10084Kdata.txt
12001Kdata.txt
12030Kdata.txt
15041Kdata.txt
15071Kdata.txt
70181Kdata.txt
71061Kdata.txt
71501Kdata.txt
79221Kdata.txt
14141Kdata.txt
14163Kdata.txt
14259Kdata.txt
14260Kdata.txt
61141Kdata.txt
61221Kdata.txt
62231Kdata.txt
64801Kdata.txt
67461Kdata.txt
67481Kdata.txt


'''

for i in range(19):
    
    temp = files[i][0]    
    file_name = 'Dataset/combine-data/'+str(temp)+'Kdata.txt'
    #print(str(temp)+'Kdata.txt')
    
    data_df = pd.read_csv(file_name, delimiter = '\t', header=None,usecols = [0,1,3,5,7])

    #convert panda to numpy array
    data = data_df.values

    #lamda values
    lamda_values = data[:,0]

    #Reflectance values
    Reflectance_values = data[:,2]
    
    Data[:,i] = Reflectance_values
    
#order of mineral: [Agglutinate, pyroxene, Plagioclase, Olivine, Ilmenite, Volcanic Glass,SMFe]

#density in g/cc
density = np.array([1.8, 3.3, 2.69, 3.3, 4.79, 2.4, 7.87])

#abundance for sample 14141(here, abundance of metallic iron not present, order different and both pyroxenes mixed)
abundance = np.zeros([19,7])

#read abundance data from file
datafile = pd.read_csv('Dataset/abundance_data_SMFe.csv',usecols=[1,2,3,4,5,6,7],header=None)
abundance = datafile.values

'''
our abundance data for 6 minerals for 19 samples

[1,2,3,4,5,6] --> [Agglutinate, pyroxene, Plagioclase, Olivine, Ilmenite, Volcanic Glass, SMFe]

       1     2     3    4    5     6    7
0   57.0  12.2  17.1  1.1  5.2   2.9
1   56.8  17.9  13.9  4.2  1.8   1.3
2   49.8  20.0  29.0  3.7  3.2   1.5
3   56.7  17.0  16.2  2.4  0.8   2.6
4   49.2  16.7  19.4  2.8  1.8   4.1
5   51.7   8.5  18.3  3.8  6.7   9.2
6   37.9  12.5  15.2  4.5  9.7  18.8
7   44.8  13.7  19.8  3.4  9.7   7.5
8   54.3   9.7  16.0  3.4  6.0   9.2
9   48.6  10.9  28.0  1.6  1.1   7.4
10  58.5  13.8  18.3  2.1  0.9   4.1
11  68.7   9.1  15.4  1.4  1.2   2.7
12  65.2  12.1  16.1  1.5  1.0   2.6
13  53.9   3.3  40.3  1.6  0.3   0.4
14  32.6   5.3  59.4  2.0  0.3   0.2
15  55.0   4.4  37.8  1.7  0.5   0.4
16  61.0   2.8  34.5  1.0  0.2   0.3
17  32.4   4.1  61.0  1.5  0.3   0.2
18  28.6   5.6  62.0  2.9  0.2   0.4

'''

#diameter has different samples for orthopyroxene. Only one needed
meanDiameter = np.array([110, 175, 11, 20, 15, 10, 1])

#incidence angle
i = math.pi/6
#emergence angle
e = 0
#phase angle
g = math.pi/6

mu = cos(e)
mu_0 = cos(i)

#Functions

def get_B(h,g):
    
    B = 1/(1 + (1/h)*tan(g/2))
    return B

def get_P(g,b,c):
    
    P = 1 + b*cos(g) + c*(1.5*cos(g)*cos(g) - .5)
    return P


def get_H(x,y):
    
    g = np.sqrt(1-y)
    r = (1-g)/(1+g)
    value = 1.0/(1-(1-g)*x*(r + (1-.5*r-r*x)*np.log((1+x)/x)))
    return value


def W_ave(index,w):
    
    #index : sample index (0-15)
    
    nume = np.zeros([7])
    deno = np.zeros([7])
    for i in range(0,7):
        nume[i] = abundance[index,i]*w[i]/(density[i]*meanDiameter[i])
        deno[i] = abundance[index,i]/(density[i]*meanDiameter[i])

    w_ave = np.sum(nume)/np.sum(deno)
    return w_ave


def Reflectance(lamda_index,m_index,w,h,b,c):
    
    #lamda_index : index of wavelength (0-461)
    #m_index : sample index (0-15)
    
    w_ave = W_ave(m_index,w)
    t1 = 1 + get_B(h,g)
    t2 = get_P(g,b,c)
    t3 = get_H(mu_0,w_ave)*get_H(mu,w_ave)
    t5 = w_ave/(4*math.pi)
    t6 = mu_0/(mu_0 + mu)
    
    #print(t1)
    ref = t5*t6*(t1*t2+t3-1)
    if math.isnan(ref):
    	print(w)
    	#print(t1)
    	#print(t2)
    	#print(t3)
    	

    return ref

# optimization 

import lmfit
from lmfit import Parameters, Minimizer
wav = 450
def residual(params):
    wavelength = wav
    #print(wavelength)
    reflectance = np.zeros(15)
    h = params['h'].value
    b = params['b'].value
    c = params['c'].value
    w = [params['w1'].value,params['w2'].value,params['w3'].value,params['w4'].value,params['w5'].value,params['w6'].value,params['w7'].value]
    for i in range(15):
        reflectance[i] = Reflectance(wavelength,i,w,h,b,c)

    reflectance_true = Data[wavelength,0:15]

    error = (reflectance - reflectance_true)*(reflectance - reflectance_true)/(2*len(reflectance))
   # print(np.sum(error))
    return error
    
params = Parameters()

w = np.zeros([461,7])

# h,b and c values for 461 wavelengths
h = np.zeros(461)
b = np.zeros(461)
c = np.zeros(461)

print("running this")
for lamda_index in range(461):

    print('wavelength = ' + str(lamda_index))
    # lambda value
    wav = lamda_index
    lamda = lamda_values[lamda_index]
    params = Parameters()
    params.add('h', value = np.random.random(), max = 1)
    params.add('b', value = np.random.random(), max = 1)
    params.add('c', value = np.random.random(), max = 1)

    temp = np.random.random(7)
    params.add('w1', value = temp[0], min = 0 , max = 1)
    params.add('w2', value = temp[1], min = 0 , max = 1)
    params.add('w3', value = temp[2], min = 0 , max = 1)
    params.add('w4', value = temp[3], min = 0 , max = 1)
    params.add('w5', value = temp[4], min = 0 , max = 1)
    params.add('w6', value = temp[5], min = 0 , max = 1)
    params.add('w7', value = temp[6], min = 0 , max = 1)

    mini = lmfit.Minimizer(residual,params)
    out = mini.minimize(method = 'nelder')
    #out = lmfit.scalar_minimize(residual, params, args = (lamda_index))


    # optimize w,h,b and c for particular lamda
    #temp_w,temp_h,temp_b,temp_c = optimize(lamda_index)

    w[lamda_index,:] = [out.params['w1'].value,out.params['w2'].value,out.params['w3'].value,out.params['w4'].value,out.params['w5'].value,out.params['w6'].value,out.params['w7'].value]
    h[lamda_index] = out.params['h']
    b[lamda_index] = out.params['b']
    c[lamda_index] = out.params['c']
    #print(lamda_index)
    #out.params.pretty_print()

    
# Save results

np.savetxt('dataw.txt',w,delimiter = ',')
np.savetxt('datah.txt',h,delimiter = ',')
np.savetxt('datab.txt',b,delimiter = ',')
np.savetxt('datac.txt',c,delimiter = ',')

# view results

hm = np.mean(h)
bm = np.mean(b)
cm= np.mean(c)
w.shape


# In[ ]:


reflectance = np.zeros([15,461])
for i in range(15):
        for j in range(461):
            reflectance[i,j] = Reflectance(j,i,w[j,:],hm,bm,cm)


# In[ ]:


from scipy.signal import savgol_filter,hilbert

for i in range(15):
    avgValues = savgol_filter(reflectance[i,:],41,1)
    plt.figure()
   # plt.plot(lamda_values,reflectance[i,:],lamda_values,Data[:,i])
    plt.plot(reflectance[i,:],'g-')
    plt.plot(avgValues,'r-')
    plt.plot(Data[:,i],'b-')
    plt.savefig(str(i) + '_avg_filter.png')
    plt.figure()
    amp_signal = hilbert(reflectance[i,:])
    amp_envelope = np.abs(amp_signal)
    plt.plot(reflectance[i,:],'g-')
    plt.plot(avgValues,'r-')
    plt.plot(Data[:,i],'b-')
    plt.savefig(str(i) + '_envelope.png')        

