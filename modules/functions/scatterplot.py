#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:34:44 2024

@author: abulellab
"""
import numpy as np
import matplotlib.pyplot as plt

#scatterplot for predicted vs actual target values
def scatterplot(y_pred_train, y_train):
    plt.scatter(y_pred_train, y_train, alpha=0.5, s=5, label="Actual vs. Predicted")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Scatterplot of Predicted vs. Actual Values")

    # Fit a linear regression trend line
    z = np.polyfit(y_pred_train, y_train, 1)
    p = np.poly1d(z)
    plt.plot(
        y_pred_train,
        p(y_pred_train),
        color="black",
        label=f"Trend line: {z[0]:.2f}x + {z[1]:.2f}",
    )

    plt.legend()
    plt.show()
    return