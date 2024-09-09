#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:23:06 2024

@author: abulellab
"""
import tensorflow as tf

# setting the learning rate scheduler
def learning_rate_scheduler(target, batch_sizes, epochs):  
    steps_per_epoch = len(target) / batch_sizes
    boundaries = [steps_per_epoch * epochs / 3, steps_per_epoch * epochs / 3 * 2]
    values = [0.003, 0.001, 0.0001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_fn)

    return optimizer
