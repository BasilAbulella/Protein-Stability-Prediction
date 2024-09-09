#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:36:41 2024

@author: abulellab
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
                             roc_curve,                          
                             auc
                            )

# Reciever operating characteristic (ROC) plot
def ROC_plot(y_actualall_train, y_predall_train):
    # convert the y_pred and y_test to 0s and 1s
    y_test_binary = (np.array(y_actualall_train) > 0).astype(int)

    # Assuming y_pred _binary and y_test_binary are now binary
    fpr, tpr, _ = roc_curve(y_test_binary, y_predall_train)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
    return