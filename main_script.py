#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:47:04 2024

@author: abulellab
"""

import numpy as np
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True) 

from modules.input_and_errors import hyperparameter_input_and_errors
from modules.model_training   import hyperparameter_model_training
from modules.evaluation       import evaluate_model
from modules.functions        import load_and_preprocess_dataset



def main():

    #1. input from user & error handling messages
    (
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
     
            ) = hyperparameter_input_and_errors()
    
    #2. loading & preprocessing the data
    (
        features, 
        target, 
        dataset
        
        ) = load_and_preprocess_dataset(
            file_name, 
            target_name, 
            scaler
            )
    
    # 3. model training for the hyperparameter run
    (
        X_train,
        X_val,
        y_train,
        y_val,
        y_actualall_train,
        y_actualall_val,
        y_pred_train,
        y_pred_val,
        y_predall_train,
        y_predall_val,
        allfolds_training_loss,
        allfolds_validation_loss,
        best_epoch_all,
        scores,
        pearson_coeff_val_all,
        dataset_model,
        history,
      
        ) = hyperparameter_model_training(
            features,
            name,
            target,
            dataset,
            n_folds,  
            hidden_layers,
            activations, 
            dropouts, 
            nodes, 
            epochs, 
            batch_sizes,
            random_states,
            npy_saving_dir,
            csv_saving_dir,
            file_description
        )
            
    # 4. evaluating the model
    
    # evaluate_model(
        
    #     X_train,
    #     X_val,
    #     y_train,
    #     y_val,
    #     y_actualall_train,
    #     y_actualall_val,
    #     y_pred_train,
    #     y_pred_val,
    #     y_predall_train,
    #     y_predall_val,
    #     allfolds_training_loss,
    #     allfolds_validation_loss,
    #     best_epoch_all,
    #     scores,
    #     pearson_coeff_val_all,
    #     dataset,
    #     dataset_model,
    #     history,
    #     name,
    #     target_name,
    #     scaler,
    #     n_folds,
    #     activations,
    #     dropouts,
    #     nodes,
    #     epochs,
    #     batch_sizes,
    #     npy_saving_dir,
    #     file_description
    # )
    
if __name__ == "__main__":
    main()

