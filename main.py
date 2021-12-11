import math

import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import math

def load_data():
    ####REPLACE THESE WITH THE ACTUAL DATASET LOADING
    pass

data_X, data_y = load_data()
tv_X, test_X, tv_y, test_y = train_test_split(data_X, data_y, 0.15)
train_X, valid_X, train_y, valid_y = train_test_split(tv_X, tv_y, 0.2)

def get_best_rfc():
    sizes = [10, 25, 100, 250, 1000]
    criteria = ["gini", "entropy"]

    max_depth = [5, 10, 50, None]

    min_valid_error = math.inf
    best_rfc = None
    for size in sizes:
        for criterion in criteria:
            for depth in max_depth:
                rfc = RandomForestClassifier(n_estimators=size, criterion=criterion, max_depth=depth)
                rfc.fit(train_X, train_y)
                valid_error = mean_squared_error(valid_y, rfc.predict(valid_X))
                if valid_error < min_valid_error:
                    min_valid_error = valid_error
                    best_rfc = rfc
    test_error = mean_squared_error(test_y, best_rfc.predict(test_X))
    return best_rfc, test_error


