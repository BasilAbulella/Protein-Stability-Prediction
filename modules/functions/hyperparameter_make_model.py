#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:20:30 2024

@author: abulellab
"""
import tensorflow as tf
from tensorflow.keras import layers


def hyperparameter_make_model(features, hidden_layer, dropout ,node, activation):
    
    input_dim = features.shape[1]
    dataset_model = tf.keras.Sequential()
    dataset_model.add(layers.Input(shape=(input_dim,)))  # Input layer
    
    for _ in range(hidden_layer):
        dataset_model.add(layers.Dense(node, activation=activation))  # Hidden layers
        dataset_model.add(layers.Dropout(dropout))  # Dropout layer
        dataset_model.add(layers.BatchNormalization())  # Batch normalization
     
    dataset_model.add(layers.Dense(1, activation='linear'))  # Output layer

    return dataset_model