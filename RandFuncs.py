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