# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 14:22:33 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 174                                                    $: Revision of the last commit

"""
Script for kNN model creation with tuned hyperparameters. Created model is saved in the current folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer


def main():

    # Import dataset.

    dataset = pd.read_csv("db_kNN_cleaned.csv")

    # Seperate features(X)/ dependant(y)

    y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)
    X = pd.get_dummies(X, drop_first=True)

    names = list(X)

    # Scale features

    scaler = StandardScaler(copy=True).fit(X)

    X = scaler.transform(X)

    # Fitting XGB regressor with parameters obtained from Gridsearch

    scores = []
    mae = []
    mse = []
    rmse = []

    # Fitting model.

    knn = neighbors.KNeighborsRegressor()

    # Define parameter grid.

    parameters = {'algorithm':['ball_tree','kd_tree',"auto"],
                  'metric':['minkowski','euclidean','manhattan','chebyshev'],
                  'weights':['uniform','distance'],
                  'leaf_size': [1,10,30,50],
                  'n_neighbors':[1,5,10,30,50],
                  'p': [1,2,3,5,10]}


    # Set Randomized Search CV Parameters (with 3 cross-fold validation and 300 iterations)

    knn_random = RandomizedSearchCV(estimator = knn, 
                                    scoring='neg_mean_squared_error', 
                                    param_distributions = parameters, 
                                    n_iter = 500, 
                                    cv = 10, 
                                    verbose=50, 
                                    random_state=42)


    #Fit model
    
    knn_random.fit(X, y)

    #Print best paramaters and corresponding score.

    print(knn_random.best_score_)
    print(knn_random.best_params_)

    #Save best paramaters and corresponding score in a csv file.

    f = open("knn_bestparameters.csv", "a")
    f.write(knn_random.best_params_)
    f.write(":")
    f.write(knn_random.best_score_)
    f.write(",")


if __name__ == '__main__':
    main()

