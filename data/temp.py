import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(1, 'C:\\Users\\basil\\Desktop\\thesis\\master-thesis\\project\\modules\\functions')
from ROC_plot import ROC_plot
from calculate_median_per_datapoint import calculate_median_per_datapoint
from scipy.stats import pearsonr
data_folder = "C:\\Users\\basil\\Desktop\\thesis\\master-thesis\\project\\data\\npy_data"

