#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:43:41 2024

@author: abulellab
"""
import os
import netron

# netron is a nice network visualizer
def netron_visualizer(dataset_model, npy_saving_dir, file_description):  
    model_path = os.path.join(
                 npy_saving_dir   + 
                 file_description + 
                 "netron_visualizer.h5"
    )

    
    dataset_model.save(model_path)
    netron.start(model_path)
    return