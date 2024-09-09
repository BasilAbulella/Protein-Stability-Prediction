#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:39:12 2024

@author: abulellab
"""

import numpy as np
import matplotlib.pyplot as plt


# plots a graph that contains all n_folds together
def cross_validation_graph(n_folds, allfolds_training_loss, allfolds_validation_loss, best_epoch_all):
    plt.figure(figsize=(12, 6))
    for i in range(n_folds):
        plt.plot(
            np.arange(len(allfolds_training_loss[i]))
            + i * len(allfolds_training_loss[i]),
            allfolds_training_loss[i],
            label=f"Training Loss - Fold {i + 1}",
            color="#1f77b4",
        )
        plt.plot(
            np.arange(len(allfolds_validation_loss[i]))
            + i * len(allfolds_validation_loss[i]),
            allfolds_validation_loss[i],
            label=f"Validation Loss - Fold {i + 1}",
            color="#ff7f0e",
        )

        # Get the best epoch for this fold
        best_epoch = best_epoch_all[i]
        # Calculate x-coordinate of the vertical line for best_epoch for this fold
        best_epoch_x = i * len(allfolds_training_loss[i]) + best_epoch
        # Plot vertical line for best_epoch for this fold
        plt.axvline(
            x=best_epoch_x,
            color="red",
            linestyle="--",
            label=f"Best Epoch - Fold {i + 1}: {best_epoch}",
        )

    plt.legend().remove()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs for Each Fold")
    plt.show()
    return