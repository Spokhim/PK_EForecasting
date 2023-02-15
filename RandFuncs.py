# Random Funcs

# Import Packages
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io 
import re
from scipy.fft import fft, fftfreq
import pycatch22
from scipy.signal import periodogram

def window_catch22(data,window_size=10,f_s=128):
    """
    Runs catch-22 on a given input.  The input is presumably an entire 10min file, but is certainly a pandas data frame.

    f_s = 128Hz by default (as per https://eval.ai/web/challenges/challenge-page/1693/overview)
    window_size = 10s by default.  
    """

    # Figure out how many elements goes into each window.
    number = int(window_size*f_s)

    # Length of the data_file
    file_len = data.shape[0]

    # Initialize list to save data per window
    li = []

    # Loop through the data file, creating windows of data and running Catch22.
    for k in np.arange(np.floor(file_len/number)):

        # Create or specify the window of data.
        # Just in case, maybe the 10min data file is completely empty or something, we need to do a try-except to move on.
        try:
            # Specify window
            window = data.iloc[int(k*number):int((k+1)*number),:]
        except:
            continue

        # Basic checks which might throw errors.  If they do, we just skip that window.
        # If array is empty skip it.
        if window.size==0:
            continue
        # If array has Nans skip it.  - This is not ideal, but just want to get something going quickly. 
        elif np.isnan(window).any().any():
            continue

        # For each column of dataframe apply catch22
        li1 = []
        for i in data.shape[1]:
            features = pycatch22.catch22_all(window[i], catch24=True, )
            li1.append(features["values"])  

        # Turn this list into an array 
        li1 = np.array(li1)
        # Add to list saving data per window
        li.append(li1)

    # Turn the list into a 3D array
    frame = np.array(li) 
    
    return frame   

# Signal quality check functions 
# Grabbed from Google Drive
# Adjusted output to return only the quality score isntead of a whole dataframe. 

def splitSignal(sig, rate, seconds, overlap, minlen):
    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            print('End of signal')
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            print('short chunk')
            split = np.hstack((split, np.zeros((int(rate * seconds) - len(split),))))
        
        sig_splits.append(split)

    return sig_splits 

def spectral_entropy(x, freq, nfft=None):    
    _, psd = periodogram(x, freq, nfft = nfft)   
    # calculate shannon entropy of normalized psd
    psd_norm = psd / np.sum(psd)
    ## turn 0 into nan to avoid divide by zero encountered in log 
    psd_norm[psd_norm == 0] = np.nan
    entropy = np.nansum(psd_norm * np.log2(psd_norm))

    result = -(entropy / np.log2(psd_norm.size))
    return result

def acc_quality_adj(acc_data, fs):
    # ratio of narrowband and broadband spectral power for 4s window
    # then consider averaged value over consecutive 10-min segments 
    rms = np.sqrt((1/3)*(acc_data['acc_x']**2 + acc_data['acc_y']**2 + acc_data['acc_z']**2))
    split_rms = splitSignal(rms.values, fs, 4, 0, 0)

    ratio = []
    for second in split_rms:
        f, psd = periodogram(second, 128, nfft = None) 
        mask1 = np.where(f>0.8)
        mask2 = np.where(f<5)
        mask3 = np.where(f<16)
        narrow = np.intersect1d(mask1, mask2)
        broad = np.intersect1d(mask1, mask3)
        Pnarrow = psd[narrow]
        Pbroad = psd[broad]
        ## set ratio to 0 to avoid invalid value encountered in double_scalars
        if Pbroad.mean() == 0:
            ratio.append(0)
        else:
            ratio.append(Pnarrow.mean()/Pbroad.mean())
    
    acc_quality = np.mean(np.array(ratio).reshape(-1, 15), axis=1)
    quality_df = pd.DataFrame(np.repeat(acc_quality,fs*60), columns= ['acc_quality'], index=acc_data.index)
    # acc_data = pd.concat([acc_data, quality_df], axis=1)
    return quality_df

def bvp_quality_adj(bvp_data, fs):
    # calculate spectral entropy of 1min data
    # (averaged over 15x 4s windows) 
    # entropy <= 0.9 is good quality 
    split = splitSignal(bvp_data['bvp'].values, fs, 4, 0, 0)

    se= []
    for second in split:
        se.append(spectral_entropy(second, fs, nfft=None))

    bvp_quality = np.mean(np.array(se).reshape(-1, 15), axis=1)
    quality_df = pd.DataFrame(np.repeat(bvp_quality,fs*60), columns= ['bvp_quality'], index=bvp_data.index)
    quality_df['bvp_quality'] = np.where(quality_df['bvp_quality'] <= 0.9, 1, 0)    # <=0.9 good quality 
    #bvp_data = pd.concat([bvp_data, quality_df], axis=1)
    return quality_df

def eda_quality_adj(eda_data, fs):
    # rate of amplitude change in concurrent 1s windows
    # changes in signal amplitude >20% increase or <10% decrease 
    split = splitSignal(eda_data['eda'].values, fs, 1, 0, 0)

    av = []
    for second in split:
        av.append(second.mean())

    eda_quality = []
    for i in range(len(av)-1):
        if av[i+1] - av[i] > av[i]*0.2:
            eda_quality.append(0)
        elif av[i+1] - av[i] < -av[i]*0.1:
            eda_quality.append(0)
        else:
            eda_quality.append(1)
    eda_quality.append(1)

    quality_df = pd.DataFrame(np.repeat(eda_quality,fs), columns= ['eda_quality'], index=eda_data.index)
    #eda_data = pd.concat([eda_data, quality_df], axis=1)
    return quality_df