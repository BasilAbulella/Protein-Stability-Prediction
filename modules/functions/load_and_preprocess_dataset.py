#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:12:21 2024

@author: abulellab
"""

import pandas as pd


def load_and_preprocess_dataset(filename, target_name, scaler):  
    # Load dataset from file
    dataset = pd.read_csv(filename, delimiter="\t", header=0)

    # Drop unnecessary columns
    dataset = dataset.drop(columns=["pdb", "wt", "mut"])

    # Perform one-hot encoding for categorical columns
    categorical_columns = ["SecStr"]
    dataset = pd.get_dummies(dataset, columns=categorical_columns)

    # Separate features (features) and target_name variable (target)
    features = dataset.drop(columns=[target_name])
    target = dataset[target_name]

    # Apply scaling to features
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return features, target, dataset