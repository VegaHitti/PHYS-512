import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
import os
import json
import scipy.integrate as sp

#Part a) -------------------------------------------------------------------------

os.chdir(r"C:\Users\Owner\Documents\Classes\F2021\PHYS 512\emacs\LOSC_Event_tutorial\LOSC_Event_tutorial")

#Creating a window, as shown in class
def make_win(n):

    x = np.linspace(-np.pi, np.pi, n)

    return 0.5 + 0.5*np.cos(x)

#Creating window with a flat period near the center
def flat_win(n, m):

    tmp = make_win(m)
    win = np.ones(n)
    mm = m//2
    win[:mm] = tmp[:mm]
    win[-mm:] = tmp[-mm:]

    return win


#The following is copy pasted from "simple_read_ligo.py"

def read_template(filename):
    
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    
    return th,tl


def read_file(filename):
    
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]
    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)
    dataFile.close()
    
    return strain,dt,utc

#See the PDF File for details on the noise model
#Also see the ft_noise() function under Part b)
#Below is an example to visualize the noise model

fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
sh,dt,utc=read_file(fname)

template_name = 'GW150914_4_template.hdf5'
th,tl=read_template(template_name)


n = len(sh)
win = flat_win(n, n//5)
sh_ft = np.fft.rfft(win*sh)
th_ft = np.fft.rfft(win*th)


n_sh_ft = np.abs(sh_ft)**2

for i in range(100):

    n_sh__ft = (n_sh_ft + np.roll(n_sh_ft, 1) + np.roll(n_sh_ft, -1))/3    

plt.figure(1)
plt.loglog(np.abs(n_sh_ft), "r")
plt.title("Hanford Noise, Event #1")
#plt.xlim(1e4, 7e4)

plt.show()


#Part b) -----------------------------------------------------------------

#Load the mapping between the events and their data

f = open("BBH_events_v3.json",)
events = json.load(f)

#This function returns the data and signal of both detectors for a given event.

def get_data(i):

    key = list(events)[i]

    f_h1 = events[key]["fn_H1"]
    f_l1 = events[key]["fn_L1"]
    s_h1, dt_h1, utc_h1 = read_file(f_h1)
    s_l1, dt_l1, utc_l1 = read_file(f_l1)

    temp = events[key]["fn_template"]
    th, tl = read_template(temp)

    return s_h1, s_l1, th, tl 
    

#Noise model from part a)

def ft_noise(x):

    x_ft = np.fft.rfft(x)
    n_ft = np.abs(x_ft)**2

    for i in range(20):

        n_ft = (n_ft + np.roll(n_ft, 1) + np.roll(n_ft, -1))/3

    return n_ft


#Whiten the fourier transform of the data

def ft_white(x, y):

   x_ft =  np.fft.rfft(x*win)
   n_ft = ft_noise(y)

   return x_ft/np.sqrt(n_ft)

#Plot the matched filter for each event and detector

for i in range(4):

    sh, sl, th, tl = get_data(i)

    plt.figure(i)
    plt.plot(np.fft.irfft(ft_white(sh, sh)*np.conj(ft_white(th, sh))),"r", label = "Hanford")
    plt.plot(np.fft.irfft(ft_white(sl, sl)*np.conj(ft_white(tl, sl))),"b", label= "Livingston")
    plt.title("Matched Filter, Event #{}".format(i+1))
    #plt.ylim(-0.025, 0.025)
    plt.legend()
    plt.show()
    

#Part c) & Part d) -------------------------------------------------------------------


#Part c)
#Take the standard deviation (ie noise) of each matched filter
#Then, calculate the signal-to-noise ratio according to this noise

#Part d)
#Take the matched filter of the template with itself, including the noise model
#Then, calculate the std of this to compute the signal-to-noise ratio

noises = []

for i in range(4):

    sh, sl, th, tl = get_data(i)

    #Creating the combined data
    shl = [np.mean([j,k]) for j,k in zip(sh, sl)]
    thl = [np.mean([j,k]) for j,k in zip(th, tl)]

    #Matched filters, as per Part b)
    h_corr = np.fft.irfft(ft_white(sh, sh)*np.conj(ft_white(th, sh)))
    l_corr = np.fft.irfft(ft_white(sl, sl)*np.conj(ft_white(tl, sl)))
    hl_corr = [np.mean([j,k]) for j, k in zip(h_corr, l_corr)]
    
    #std to estimate noise
    nh = np.std(h_corr[-30000:])
    nl = np.std(l_corr[-30000:])
    nhl = np.std(hl_corr[-30000:])
    
    #Signal-to-noise ratio with noise estimate
    rh = np.max(np.abs(h_corr))/nh
    rl = np.max(np.abs(l_corr))/nl
    rhl = np.max(np.abs(hl_corr))/nhl

    #Noise according to noise model
    Nh = ft_noise(sh*win)
    Nl = ft_noise(sl*win)
    Nhl = ft_noise(shl*win)

    #Take the matched filter of the template with itself, including the noise model
    h = np.fft.irfft(ft_white(th, sh)*np.conj(ft_white(th, sh))/Nh)
    l = np.fft.irfft(ft_white(tl, sl)*np.conj(ft_white(tl, sl))/Nl)
    hl = np.fft.irfft(ft_white(thl, shl)*np.conj(ft_white(thl, shl))/Nhl)

    #Take the std
    NH = np.std(h)
    NL = np.std(l)
    NHL = np.std(hl)

    #Signal-to-noise ratio, according to the noise model
    Rh = np.sqrt(np.max(np.abs(h))/NH)
    Rl = np.sqrt(np.max(np.abs(l))/NL)
    Rhl = np.sqrt(np.max(np.abs(hl))/NHL)
       
    noises.append([nh, nl, nhl, NH, NL, NHL])

    #print("The estimated noise for event {} is {} for Hanford and {} for Livingston".format(i+1, nh, nl))
    #print("The analytical noise for event {} is {} for Hanford and {} for Livingston".format(i+1, Nh, Nl))
    #print("The estimated signal-to-noise ratio for event {} is {} for Hanford, {} for Livingston and {} for both".format(i+1, rh, rl, rhl))
    #print("The analytical signal-to-noise ratio for event {} is {} fo Hanford, {} for Livingston and {} for both".format(i+1, Rh, Rl, Rhl))
   


#Part e) ---------------------------------------------------------------------

for i in range(4):

    sh, sl, th, tl = get_data(i)

    #Matched filters in Fourier space
    mf_ft_h = ft_white(sh,sh)*np.conj(ft_white(th,sh))
    mf_ft_l = ft_white(sl,sl)*np.conj(ft_white(tl,sl))

    #Cumulative integral of the matched filters in Fourier space
    max_h = np.max(np.abs(sp.cumtrapz(mf_ft_h)))
    max_l = np.max(np.abs(sp.cumtrapz(mf_ft_l)))

    indh = (np.abs((np.abs(mf_ft_h)-0.5*max_h))).argmin()
    indl = (np.abs((np.abs(mf_ft_l)-0.5*max_l))).argmin()
    
    print(max_h, max_l)
    print(indh, indl)

    fig, ax = plt.subplots(1, 2, figsize = (12,5))
    
    ax[0].plot(mf_ft_h[:20000], "m")
    ax[0].plot(mf_ft_h[:indh], "c")
    ax[0].set_title("Weights vs. Frequency for Event #{}, Hanford".format(i+1))
    
    ax[1].plot(mf_ft_l[:20000], "m")
    ax[1].plot(mf_ft_l[:indl], "c")
    ax[1].set_title("Weights vs. Frequency for Event #{}, Livingston".format(i+1))    
    
    plt.show()


#Part f) ------------------------------------------------------------------------

#It's all in the PDF

   












