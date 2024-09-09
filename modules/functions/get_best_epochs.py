#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:28:36 2024

@author: abulellab
"""
import numpy as np

# returns the epoch number at which the lowest loss was recorded
def get_best_epochs(history):
    best_epoch = np.argmin(history.history["loss"])
    best_val_epoch = np.argmin(history.history["val_loss"])
    return best_epoch, best_val_epoch