# Uses Auto-sklearn to train a classifier on the data

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

import autosklearn.classification
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn import metrics

# Things to change:

# Import the Data
df = pd.read_pickle("Patient_1110.pkl")
# Specify Columns to Use as Input (in addition to the time-stamp) noting that we'll take all of the catch22 features.
cols = ["acc_mag","acc_theta","acc_phi",]

##################################################################################################################

feature_names = np.loadtxt("Catch22_Featurenames",dtype=str)
col_names = [i + "_" + j for i in cols for j in feature_names]
col_names = col_names + ['time_of_day'] # Add the quality measurements later.

# Get time-of-day from timestamp
df['time_of_day'] = df['utc_timestamp'][0].hour + df['utc_timestamp'][0].minute/60 

# Perform train-test split. Just Split it across time 3:1.  For patient 1110, this ensures one seizure in test set.
# And need to drop time-stamp. As datetime64 is unsupported 
training_data = df[col_names].iloc[0:int(df.shape[0]*0.75),:]
testing_data = df[col_names].iloc[int(df.shape[0]*0.75):,:]
training_target = df['label'].iloc[0:int(df.shape[0]*0.75)]
testing_target = df['label'].iloc[int(df.shape[0]*0.75):]

# Initialize the auto-sklearn classifier
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=180,
                                                           per_run_time_limit=30,
                                                           n_jobs=-1,
                                                           memory_limit = 102400000)  # This needs to be set or else you may error out. 

# Fit the classifier on the training data
automl.fit(training_data, training_target)

# Score the classifier on the testing data
print("Accuracy: ", automl.score(testing_data, testing_target))

# Save the model
dump(automl, 'Fast_automl_acc.joblib')