#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:25:26 2024

@author: abulellab
"""

def model_evaluation(dataset_model, X_train, X_val, y_train, y_val, scores, allfolds_training_loss, allfolds_validation_loss ,history):  
    # calculate loss    
    loss = dataset_model.evaluate(X_train, y_train)  
    # append the loss to the loss list
    scores.append(loss)  
    
    # Make predictions on the test data, flatten was done to reduce the 
    # dimentions so that it is compatible with later functions
    y_pred_train = dataset_model.predict(
        X_train
    ).flatten() 

    # we do the same for the validation set
    loss_val = dataset_model.evaluate(
        X_val, y_val
    )  
    scores.append(loss_val)
    y_pred_val = dataset_model.predict(X_val).flatten()


    # changing from dataframe to array, in order to be
    # compatible with the pearson function
    y_train = (
        y_train.values
    )  
    y_val = y_val.values  # 

    allfolds_training_loss.append(history.history["loss"])
    allfolds_validation_loss.append(history.history["val_loss"])

    return y_pred_train, y_pred_val, y_train, y_val