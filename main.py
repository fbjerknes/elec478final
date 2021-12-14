import math

import numpy as np
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import matplotlib.pyplot as plt
import math

items1 = ["Molecular Weight", "Targets", "Bioactivities", "AlogP", "Polar Surface Area", "HBA", "HBD",
                            "#RO5 Violations", "#Rotatable Bonds", "QED Weighted", "CX Acidic pKa", "CX Basic pKa",
                            "CX LogP", "CX LogD", "Aromatic Rings", "Heavy Atoms", "Molecular Species"]
items2 = ["Molecular Weight", "Targets", "Bioactivities", "AlogP", "Polar Surface Area", "HBA", "HBD",
                            "#RO5 Violations", "#Rotatable Bonds", "QED Weighted",
                            "CX LogP", "CX LogD", "Aromatic Rings", "Heavy Atoms", "Molecular Species"]

data_X = None
data_y = None
train_X = None
train_y = None
valid_X = None
valid_y = None
test_X = None
test_y = None

def load_data(it):
    global data_X, data_y
    if it:
        ite = items1
    else:
        ite = items2
    dfx = pd.read_csv('drugdata.csv', delimiter=";")
    dfx = dfx.filter(items=ite)
    dfx = dfx.mask(dfx.eq("None")).dropna()
    dfy = dfx["Molecular Species"]
    ite.remove("Molecular Species")
    dfx = dfx.filter(items=ite)
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

    data_X = dfx.values
    data_y = y_data



def set_params():
    global test_X, test_y, train_y, train_X, valid_X, valid_y
    tv_X, test_X, tv_y, test_y = train_test_split(data_X, data_y, test_size=0.15)
    train_X, valid_X, train_y, valid_y = train_test_split(tv_X, tv_y, test_size=0.2)

def logistic_features():
    lr = LogisticRegression(random_state=0).fit(train_X, train_y)
    return accuracy_score(test_y, lr.predict(test_X)),

def naive_bayes_features():
    nb1 = GaussianNB().fit(train_X, train_y)
    nb2 = BernoulliNB().fit(train_X, train_y)
    # nb3 = MultinomialNB().fit(train_X, train_y)
    return accuracy_score(test_y, nb1.predict(test_X)), accuracy_score(test_y, nb2.predict(test_X))


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
    test_error = accuracy_score(test_y, best_rfc.predict(test_X))
    return best_rfc, test_error



def run(control):
    if (control):
        i = items1
        st = "Control"
    else:
        i = items2
        st = "Experiment"
    load_data(control)
    set_params()
    print(control)
    print("_________")
    lscore = logistic_features()
    print(lscore)
    nbscore0 = naive_bayes_features()[0]
    nbscore1 = naive_bayes_features()[1]
    print(nbscore0)
    print(nbscore1)
    rfc, rfcscore = get_best_rfc()
    print(rfcscore)
    print(rfc.feature_importances_)
    plt.bar(i, rfc.feature_importances_)
    plt.title("Feature Importances: " + st)
    plt.xticks(rotation=65, fontsize=6)
    plt.show()
    scores = [lscore, nbscore0, nbscore1, rfcscore]
    scorestrings = ["Logistic", "Gaussian NB", "Bernoulli NB", "Random Forest"]
    plt.bar(scorestrings, scores)
    plt.title("Model Accuracy: " + st)
    plt.show()
    print("_________")

run(True)
run(False)



