# Combines all the data and creates the testing set of features.

# Import Packages
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io 
import re
from scipy.fft import fft, fftfreq
from pathlib import Path

import pycatch22
import RandFuncs

# Stuff that you might need to change: (aka paths to the data)

patient = "2002"
#start_root = os.getcwd()+'/Ignore/Output/'
start_root = '/data/gpfs/projects/punim1887/msg-seizure-forecasting/data/' 
# root is the folder/directory of the patient.  
root = start_root + 'train/' + patient + '/'
# Load the labels
labels = pd.read_csv(start_root + 'train_labels.csv')

f_s=128
########################


# Get all parquet files in the directory and subdirectory. 
# https://stackoverflow.com/questions/2909975/python-list-directory-subdirectory-and-files
files = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files if name[-8:]==".parquet"]
# Sort Path by Ascending Order
files.sort(reverse=False)

# As the patient "only has 32GB of data", we could probably load it all into memory.

# Iterate through all the files and turn it into a single dataframe
# Load all the DataFrames into an Empty List and Concat at end for efficiency.  
# Initialise an empty List
li = []

for file_path in files:

    # First load one of the parquet files of the patient
    data = pd.read_parquet(file_path,engine='pyarrow')
    # Change first column to datetime format
    data['utc_timestamp'] = pd.to_datetime(data['utc_timestamp'], utc=True, unit='s')

    # Add in the training labels. 
    # No clue which ones are the updated ones, so just use the old ones. 
    # Get rid of the os.cwd() component of the path to match it with the labels csv
    path = file_path.replace(start_root + 'train/', '')
    label = labels[['label']].loc[labels['filepath'] == str(path)].values[0]

    data['label'] = np.repeat(label, repeats = data.shape[0])

    # Perform NAN treatment later ###########################################################

    # Append to list
    li.append(data)

# Convert to df
data = pd.concat(li,axis=0,ignore_index=True)
# Sort by the time
data = data.sort_values(by="utc_timestamp")

# Want to transform from Cartesian to Polar Coordinates

# First get the magnitude of the acceleration
# For some reason acc_mag is different from the calcualted magnitude.  By roughly 1.  My guess is that their mag was calculated after a normalisation step which removed the mean of 1g. 
# print(data["acc_mag"] - np.sqrt(data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2))
# Therefore, replace it with a calculated one
data["acc_mag"] = np.sqrt(data["acc_x"]**2 + data["acc_y"]**2 + data["acc_z"]**2)

# Then get theta
data["acc_theta"] = np.arctan2(data["acc_y"],data["acc_x"])

# Then get phi
data["acc_phi"] = np.arccos(data["acc_z"]/data["acc_mag"])

# Add in the data_quality features
data["acc_quality"] = RandFuncs.acc_quality_adj(data, f_s)
data["bvp_quality"] = RandFuncs.bvp_quality_adj(data, f_s)
data["eda_quality"] = RandFuncs.eda_quality_adj(data, f_s)

# Time to Apply Catch22

window_size = 10
f_s = 128

# Specify columns to use
cols = ["acc_mag","acc_theta","acc_phi","bvp","eda","hr"]
# Extract relevant columns
#data = data[cols + ["utc_timestamp"]]   

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
    # elif np.isnan(window).any().any():
    #     continue

    # For each column of dataframe apply catch22
    li1 = []
    for i in cols:
        features = pycatch22.catch22_all(window[i], catch24=True, )
        li1.append(features["values"])  

    # Flatten the list of lists
    li1 = [item for sublist in li1 for item in sublist]
    # Record the utc_timestamp of the start of each window
    li1.append(window['utc_timestamp'].iloc[0])    
    # Record the label of the start of each window
    li1.append(window['label'].iloc[0])
    # Record the three quality metrics at the start of each window
    li1.append(window['acc_quality'].iloc[0])
    li1.append(window['bvp_quality'].iloc[0])
    li1.append(window['eda_quality'].iloc[0])

    # Add to list saving data per window
    li.append(li1)

# Turn the list into the appropriate pd.DataFrame
# Create column names
feature_names = np.loadtxt("Catch22_Featurenames",dtype=str)
col_names = [i + "_" + j for i in cols for j in feature_names]
col_names = col_names + ["utc_timestamp"] + ["label"] + ["acc_quality"] + ["bvp_quality"] + ["eda_quality"]

# Create dataframe
df = pd.DataFrame(li,columns=col_names)

# Save it as a pickle. 
df.to_pickle("Patient_" + patient + "_Data.pkl")