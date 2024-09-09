#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:42:19 2024

@author: abulellab
"""
import os
import sweetviz as sv


# sweetviz report for the dataset
def sweetviz_report(dataset, npy_saving_dir, file_description):
    dataset_report = sv.analyze(dataset)
    # Save the report as an HTML file
    model_path = os.path.join(
                 npy_saving_dir   + 
                 file_description + 
                 "sweetviz_report"
    )
    
    dataset_report.show_html(model_path)             
    return