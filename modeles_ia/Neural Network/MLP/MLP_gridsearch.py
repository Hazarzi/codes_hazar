# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 10:32:27 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 100                                                    $: Revision of the last commit

"""
Script for MLP model optimization using GridSearch.
"""

import pandas as pd
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.model_selection import GridSearchCV


def main():

    #import dataset

    dataset = pd.read_csv("db_kNN_cleaned.csv")

    y = dataset.TMP

    X = dataset.drop(["TMP"], axis=1)

    # One hot encoding

    X = pd.get_dummies(X, drop_first=True)

    # Scale features

    X = StandardScaler().fit_transform(X)

    y = y.values
    X = X.values

    # Define alpha and learning rate parameters to try.
    alphas = np.logspace(-1, 0, num=10)
    rates = np.logspace(-4, -2, num = 5)

    mlp = MLPRegressor(random_state = 42)

    # Define parameter grid to search for beset parameters.

    parameters = {'hidden_layer_sizes': [(64,64,),(32,32),(128,),(64,),(32,32,32)],
                  'activation':['tanh','logistic','relu'],
                  'solver': ['lbfgs','adam','sgd'], 'max_iter': [500, 1000, 1500], 'alpha' : 10.0 ** -np.arange(1, 10),
                  'learning_rate_init':[0.0001, 0.001, 0.01, 0.1]}

    # Set up 3 fold grid search.

    mlp_random = RandomizedSearchCV(estimator=mlp, scoring='neg_mean_squared_error', param_distributions=parameters,
                                    n_iter=400, cv=4, verbose=2, random_state=42)

    mlp_grid.fit(X, y)

    print(mlp_random.best_params_)

    #Save best paramaters and corresponding score in a csv file.

    f = open("mlp_bestparameters.csv", "a")
    f.write(knn_random.best_params_)
    f.write(":")
    f.write(knn_random.best_score_)
    f.write(",")

if __name__ == '__main__':
    main()
