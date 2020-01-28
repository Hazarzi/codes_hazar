# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                    $: Revision of the last commit

"""
Script for kNN model optimization using GridSearch.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


def main():

    #import dataset

    dataset = pd.read_csv("db_kNN_cleaned.csv")

    # One hot encoding

    y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)
    X = pd.get_dummies(X, drop_first=True)

    names = list(X)

    # scaling features

    scaler = StandardScaler(copy=True).fit(X)

    X = scaler.transform(X)

    # Fitting XGB regressor with parameters obtained from Gridsearch

    scores = []
    mae = []
    mse = []
    rmse = []

    #Fitting SVR

    model = SVR()

    #Define hyperparameter space.

    parameters = {'kernel':['rbf','sigmoid','linear'],
                  'gamma': ['auto','scale'],
                  'C': [1, 10 ,30, 50, 100],
                  'epsilon': [0.01, 0.05, 0.1, 0.5, 1],
                  'tol':[0.0001, 0.001, 0.01, 0.1, 1]}

    # Setup randomsearch.

    svr_random = RandomizedSearchCV(estimator = svr,
                                    scoring= 'neg_mean_squared_error',
                                    param_distributions = parameters,
                                    refit= True,
                                    n_iter = 300,
                                    cv = 4,
                                    verbose=50,
                                    random_state=42)

    # Start randomsearch.

    svr_random.fit(X, y)

    print(svr_random.best_params_)

    #Save best paramaters and corresponding score in a csv file.

    f = open("svr_bestparameters.csv", "a")
    f.write(knn_random.best_params_)
    f.write(":")
    f.write(knn_random.best_score_)
    f.write(",")

if __name__ == '__main__':
    main()

