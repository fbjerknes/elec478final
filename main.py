import math

import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import math

items1 = ["Molecular Weight", "Targets", "Bioactivities", "AlogP", "Polar Surface Area", "HBA", "HBD",
                            "#RO5 Violations", "#Rotatable Bonds", "QED Weighted", "CX Acidic pKa", "CX Basic pKa",
                            "CX LogP", "CX LogD", "Aromatic Rings", "Heavy Atoms", "Molecular Species"]
items2 = ["Molecular Weight", "Targets", "Bioactivities", "AlogP", "Polar Surface Area", "HBA", "HBD",
                            "#RO5 Violations", "#Rotatable Bonds", "QED Weighted",
                            "CX LogP", "CX LogD", "Aromatic Rings", "Heavy Atoms", "Molecular Species"]
def load_data(it):
    if it:
        ite = items1
    else:
        ite = items2
    dfx = pd.read_csv('drugdata.csv', delimiter=";")
    dfx = dfx.filter(items=ite)
    dfx = dfx.mask(dfx.eq("None")).dropna()
    dfy = dfx["Molecular Species"]
    dfx = dfx.filter(items=ite)
    print(dfx)
    y_data = []
    for row in dfy:
        if row == "NEUTRAL":
            y_data.append(0)
        elif row == "BASE":
            y_data.append(1)
        elif row == "ACID":
            y_data.append(2)
        elif row == "ZWITTERION":
            y_data.append(3)
        else:
            y_data.append(-1)
    return dfx.values[:250000], y_data[:250000]

data_X, data_y = load_data()
tv_X, test_X, tv_y, test_y = train_test_split(data_X, data_y, test_size=0.15)
train_X, valid_X, train_y, valid_y = train_test_split(tv_X, tv_y, test_size=0.2)




def get_best_rfc():
    sizes = [10, 25, 100]
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
                print(size)
    test_error = mean_squared_error(test_y, best_rfc.predict(test_X))
    return best_rfc, test_error

rfc, error = get_best_rfc()

print(error)
print(rfc.feature_importances_)
print(pca_features())

