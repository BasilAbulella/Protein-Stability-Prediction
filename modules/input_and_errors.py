#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:31:19 2024

@author: abulellab
"""
import  sys
from    sklearn.preprocessing import (
        StandardScaler,
        RobustScaler,
        MinMaxScaler,
        PowerTransformer,
)

def hyperparameter_input_and_errors():
    #define hyperparameter to test, parameters takes around 20480 mins or approximately 2 weeks
    try:
        # Assign command-line arguments to variables
           
        name          = sys.argv[1]
        target_name   = sys.argv[2]
        scaler        = sys.argv[3]
        n_folds       = int(sys.argv[4])
        
        hidden_layers = [3]          
        activations   = ['LeakyReLU']
        dropouts      = [0, 0.1, 0.2]
        nodes         = [12, 16, 20, 24]
        epochs        = [200]
        batch_sizes   = [32]
        random_states = [42]

        # saving directories, need to be changed if the program was moved.
        file_dir          = ("/home/people/abulellab/masterarbeit/project/files/")
        npy_saving_dir    = ("/home/people/abulellab/masterarbeit/project/data/npy_data/")
        csv_saving_dir    = ("/home/people/abulellab/masterarbeit/project/data/csv_data/")
        file_name         = (file_dir + name)
        file_description  = (str(name)          + "_" + 
                             str(target_name)   + "_" + 
                             str(scaler)        + "_" +
                             "results")

        # 2. error handling
        # Mapping of scaler names to scaler classes
        scalers = {
            "power"     : PowerTransformer(),
            "standard"  : StandardScaler(),
            "minmax"    : MinMaxScaler(),
            "robust"    : RobustScaler(),
        }

        if scaler not in scalers:
            print(
                "Invalid scaler name on argument 3, Available scalers: 'power', 'standard', 'minmax', 'robust'"
            )
            sys.exit(1)

        scaler = scalers[scaler]

    except:
        sys.exit(1)

    return (
        name, 
        target_name, 
        scaler, 
        n_folds, 
        hidden_layers,
        activations, 
        dropouts, 
        nodes, 
        epochs, 
        batch_sizes,
        random_states,
        file_dir,
        file_name,
        npy_saving_dir,
        csv_saving_dir,
        file_description
            )