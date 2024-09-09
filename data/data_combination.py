#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:58:58 2024

@author: abulellab
"""

import os
import sys
import numpy as np
import pandas as pd

# specify the directory of functions
sys.path.insert(1, 'C:\\Users\\basil\\Desktop\\thesis\\master-thesis\\project\\modules\\functions')
from ROC_plot import ROC_plot
from calculate_median_per_datapoint import calculate_median_per_datapoint

from scipy.stats import pearsonr

# Data directory
data_folder = "C:\\Users\\basil\\Desktop\\thesis\\master-thesis\\project\\data\\npy_data"
# Gather .npy files in the directory
data_files  = [f for f in os.listdir(data_folder) if f.endswith('.npy') and f != '.npy']

# Initialize lists to hold data
y_actualall_train    = []
y_predall_train      = []
y_predall_val_sorted = []
target               = []


# Load data from each file
for file in data_files:
    file_path = os.path.join(data_folder, file)
    # Check if the path is a file
    if os.path.isfile(file_path):  
        data = np.load(file_path, allow_pickle=True).item()
        
        # Append data to respective lists 
        y_actualall_train.append   (data.get('y_actualall_train')) 
        y_predall_train.append     (data.get('y_predall_train')) 
        y_predall_val_sorted.append(data.get('y_predall_val_sorted')) 
        target.append              (data.get('target'))

# Calculate median for each position for all arrays
y_actualall_train_median    = calculate_median_per_datapoint(y_actualall_train)
y_predall_train_median      = calculate_median_per_datapoint(y_predall_train)
y_predall_val_sorted_median = calculate_median_per_datapoint(y_predall_val_sorted)
target                      = calculate_median_per_datapoint(target)

# Calculate Pearson correlation coefficients
pearson_coeff_train_allruns, _ = pearsonr(y_predall_train_median ,
                                           y_actualall_train_median)
print("pearson_coeff_train for all runs:", 
      pearson_coeff_train_allruns)

pearson_coeff_val_allruns, _ = pearsonr(y_predall_val_sorted_median ,
                                          target)
print("pearson_coeff_val for all runs:", 
      pearson_coeff_val_allruns)

#ROC plot for all runs
ROC_plot(y_actualall_train_median, y_predall_train_median)

# # Convert each array in y_predall_val_sorted into a Series
consensus_data = [pd.Series(arr) for arr in y_predall_val_sorted]

# Concatenate the Series into a DataFrame
consensus_data                  = pd.concat(consensus_data, axis=1)
consensus_data['median']        = y_predall_val_sorted_median
consensus_data['std_deviation'] = consensus_data.std(axis=1)
consensus_data['target']        = target


# Save the DataFrame to a CSV file
consensus_data.to_csv('consensus_data.csv', index=False, sep = '\t')