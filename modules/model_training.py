#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:47:04 2024

@author: abulellab
"""

import os

import numpy      as np
import tensorflow as tf
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

from itertools               import product
from sklearn.model_selection import KFold
from scipy.stats             import pearsonr
from modules.functions       import (
                                    hyperparameter_make_model,
                                    learning_rate_scheduler,
                                    model_evaluation,
                                    get_best_epochs
                                    )

def hyperparameter_model_training(
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

):
    
    kf = KFold(
        n_splits=n_folds, shuffle=True, random_state=11)

    # lists to collect data from the nfold loops
    scores                                     = []  
    allfolds_training_loss                     = []  
    allfolds_validation_loss                   = []  
    y_predall_train                            = []
    y_predall_val                              = []
    y_actualall_train                          = []
    y_actualall_val                            = []
    pearson_coeff_val_all                   = np.array([])
    y_predall_val_sorted   = np.zeros(len(target), dtype=float) 
    best_epoch_all                          = np.array([])  

    #save the data to a file.
    output_file_path = os.path.join(csv_saving_dir + file_description + '.csv')
    outfile=open(output_file_path,'w')

    ll=[
        'Hidden_Layers',
        'Activation',
        'Dropout',
        'Nodes',
        'Batch_Size',
        'Best Epoch per fold',
        'Pearson_Coefficient_train',
        'Pearson_Coefficient_val'
        ]

    outfile.write('\t'.join(ll)+'\n')
    outfile.flush()   
    
    # hyperparameter loop where we Iterate over each parameter
    for (hidden_layer, activation, dropout, node, epoch, batch_size, rand) in product(
         hidden_layers, activations, dropouts, nodes, epochs, batch_sizes, random_states
        ):

        # cross validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rand)
        
        for train_index, val_index in kf.split(features):  
            # Split the data into train and validation sets for this fold

            X_train, X_val = (features.iloc[train_index], 
                              features.iloc[val_index])   
            y_train, y_val = (target.iloc[train_index],      
                              target.iloc[val_index])
                     
            # make model
            dataset_model = hyperparameter_make_model(
                features, 
                hidden_layer, 
                dropout ,
                node, 
                activation,
                ) 
            
            #setting up custom optimizer
            optimizer = learning_rate_scheduler(target, batch_size, epoch) 
            
            # Compile the model
            dataset_model.compile(
                loss=tf.keras.losses.MeanAbsoluteError(), optimizer=optimizer) 
            
            # Fitting the model & used the splitted x_val & y_val to avoid splitting twice
            
            history = dataset_model.fit(  
                X_train,
                y_train,
                epochs=epoch,
                batch_size=batch_size,
                validation_data=(X_val, y_val))  

            (
                y_pred_train, 
                y_pred_val, 
                y_train, 
                y_val
                ) = model_evaluation(
                    dataset_model, 
                    X_train, X_val, 
                    y_train, y_val, 
                    scores, 
                    allfolds_training_loss, 
                    allfolds_validation_loss,
                    history
                    ) 
               
            # collecting & generating the data across all cross validations
            y_predall_train += list(y_pred_train.flatten())  
            y_actualall_train += list(y_train)
            
            y_predall_val += list(y_pred_val.flatten())
            y_actualall_val += list(y_val)        
              
            best_epoch, best_val_epoch = get_best_epochs(history)
            best_epoch_all = np.append(best_epoch_all, best_val_epoch)
            
            # re-sorting y_val
            val_list = list (y_pred_val.flatten())
            for i in range(len(val_index)):
                y_predall_val_sorted[val_index[i]] = val_list[i]
            
            # Pearson correlation coefficient 
            pearson_coeff_train, _ = pearsonr(
                y_pred_train,
                y_train)
            
            # Pearson correlation coefficient for the val_set
            pearson_coeff_val, _ = pearsonr(
                y_pred_val, 
                y_val)  
            
            # appending val results for every fold
            pearson_coeff_val_all = np.append(
                pearson_coeff_val_all, 
                pearson_coeff_val)
            
            print("pearson_coeff_train per fold: ",pearson_coeff_train)
            print("pearson_coeff_val per fold  : " ,pearson_coeff_val)
                        
            print("\n fold complete \n")

       #pearson for train and val sets for folds combined
        pearson_coeff_train_allfolds, _ = pearsonr(y_predall_train ,
                                                   y_actualall_train)
     
        pearson_coeff_val_allfolds, _ = pearsonr(y_predall_val ,
                                                 y_actualall_val)
        
        print("pearson_coeff_train across all folds:", 
             pearson_coeff_train_allfolds)
        print("pearson_coeff_val across all fold:   ", 
             pearson_coeff_val_allfolds)                           
        
        #.npy file creation
        saved_data = {
            'y_actualall_train':                     y_actualall_train,
            'y_predall_train':                       y_predall_train,        
            'y_actualall_val':                       y_actualall_val,
            'y_predall_val_sorted':                  y_predall_val_sorted,
            'target':                                target,
            }
        
        npy_file_description = f"model_data_{name}_{hidden_layer}_{activation}_{dropout}_{node}_{epoch}_{batch_size}_{rand}"
        np.save((npy_saving_dir + npy_file_description + ".npy"), saved_data)
        
        #results.append(result)             
        ll=[str(hidden_layer),
            str(activation),
            str(dropout),
            str(node),
            str(batch_size),
            str(best_epoch_all),
            str(pearson_coeff_train_allfolds),
            str(pearson_coeff_val_allfolds),
            ]
        outfile.write('\t'.join(ll) + '\n')
        outfile.flush()
        
        
        #reset values
        scores                                     = []  
        allfolds_training_loss                     = []  
        allfolds_validation_loss                   = []  
        y_predall_train                            = []
        y_predall_val                              = []
        y_actualall_train                          = []
        y_actualall_val                            = []
        pearson_coeff_val_all                   = np.array([])
        y_predall_val_sorted   = np.zeros(len(target), dtype=float) 
        best_epoch_all                          = np.array([])  

        print("\n run complete \n")

        
    print("program complete, generating evaluation data \n")
        
    return (
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
        )
