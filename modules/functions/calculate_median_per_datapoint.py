#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:51:47 2024

@author: abulellab
"""
import numpy as np

def calculate_median_per_datapoint(list_of_arrays):
    # Function to calculate median for each position across multiple arrays
    median_per_datapoint = []
    # Iterate over each position
    for i in range(len(list_of_arrays[0])):
        # Extract values at position i from all arrays
        values_at_position = [arr[i] for arr in list_of_arrays]
        # Calculate the median for values at position i
        median_at_position = np.median(values_at_position)
        # Append the median to the list
        median_per_datapoint.append(median_at_position)

    # Convert the list to a numpy array
    return np.array(median_per_datapoint)