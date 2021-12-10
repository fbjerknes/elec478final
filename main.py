import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

def load_data():
    ####REPLACE THESE WITH THE ACTUAL DATASET LOADING
    pass

sizes = [10, 25, 100, 250, 1000]
criteria = ["gini", "entropy"]
data_X, data_y = load_data()
max_depth = [5, 10, 50, None]
tv_X, test_x, tv_y, test_y = train_test_split(data_X, data_y, 0.15)
train_X, valid_x, train_y, valid_y = train_test_split(tv_X, tv_y, 0.2)
for size in sizes:
    for criterion in criteria:
        for depth in max_depth:
            rfc = RandomForestClassifier(n_estimators=size, criterion=criterion, max_depth=depth)
            rfc_fit = rfc.fit(train_X, train_y)