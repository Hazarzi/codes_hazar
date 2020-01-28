# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 14:22:33 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 174                                                    $: Revision of the last commit

"""
Script for RF model optimization using RandomGridSearch.
"""

import pandas as pd
import numpy as np
import time

from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def main():
    
    # Import dataset

    dataset = pd.read_csv("db_kNN_cleaned.csv")

    # One hot encoding

    dataset = pd.get_dummies(dataset)

    # Seperate features(X)/ dependant(y)

    y = dataset.TMP

    X = dataset.drop("TMP", axis=1)


    transformer = StandardScaler().fit(X)

    X = transformer.transform(X)

    # Fitting XGB regressor

    # Number of trees in random forest

    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]

    # Number of features to consider at every split

    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree

    max_depth = [int(x) for x in np.linspace(10, 200, num = 10)]
    max_depth.append(None)

    # Minimum number of samples required to split a node

    min_samples_split = [2, 3, 5, 10]

    # Minimum number of samples required at each leaf node

    min_samples_leaf = [1, 2, 4, 6]

    # Method of selecting samples for training each tree

    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}


    rf = RandomForestRegressor(random_state = 42)

    
    # Random search of parameters, using 4 fold cross validation,
    # search across 300 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, scoring='neg_mean_squared_error',  param_distributions = random_grid, n_iter = 300, cv = 4, verbose=2, random_state=42)

    # Fit the random search model

    rf_random.fit(X,y)

    rf_random.best_params_

    #Save best paramaters and corresponding score in a csv file.

    f = open("rf_bestparameters.csv", "a")
    f.write(knn_random.best_params_)
    f.write(":")
    f.write(knn_random.best_score_)
    f.write(",")

if __name__ == '__main__':
    main()


