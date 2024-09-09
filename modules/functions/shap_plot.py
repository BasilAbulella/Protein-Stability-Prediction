#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:40:53 2024

@author: abulellab
"""
import tensorflow as tf
import shap


def shap_plot(dataset_model, X_train, X_val):
    model_func = tf.function(
        dataset_model
    )  
    # Convert the TensorFlow model's predict method to a callable TensorFlow function
    # Create a DeepExplainer object
    explainer = shap.Explainer(model_func, X_train) 
    # Compute SHAP values
    shap_values = explainer.shap_values(X_val)  
    # Visualize the SHAP values
    shap.summary_plot(shap_values, X_val)  
    return