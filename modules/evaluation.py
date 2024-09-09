#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:47:19 2024

@author: abulellab
"""

from modules.functions import (scatterplot,
                               ROC_plot,
                               cross_validation_graph,
                               shap_plot,
                               sweetviz_report,
                               netron_visualizer
                               )


def evaluate_model(
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
    dataset,
    dataset_model,
    history,
    name,
    target_name,
    scaler,
    n_folds,
    activation,
    dropout,
    nodes,
    epochs,
    batch_sizes,
    npy_saving_dir,
    file_description
):
                   
    mean_score = sum(scores) / len(scores)

    # Print the evaluation results
    # print("Mean Absolute Error (MAE) for each fold:", scores)
    print("Mean MAE across all folds:", mean_score)

    # 17. overall visualization
    scatterplot(y_pred_train, y_train)
    cross_validation_graph(n_folds, allfolds_training_loss, allfolds_validation_loss, best_epoch_all)
    ROC_plot(y_actualall_train, y_predall_train)
    shap_plot(dataset_model, X_train, X_val)
    sweetviz_report(dataset, npy_saving_dir, file_description)
    netron_visualizer(dataset_model, npy_saving_dir, file_description)
    
    print("\n Evaluation results are complete, press ctrl + C to stop the program")

pass
