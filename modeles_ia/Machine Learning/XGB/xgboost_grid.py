# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 14:22:33 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 174                                                    $: Revision of the last commit

"""
Script for XGB model creation with tuned hyperparameters. Created model is saved in the current folder.
"""

import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split    

import time


def main():

    # Import dataset

    dataset = pd.read_csv("db_kNN_cleaned.csv")

    # seperate features(X)/ dependant(y)

    y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)
    X = pd.get_dummies(X, drop_first=True)

    names = list(X)

    # scaling features

    scaler = StandardScaler(copy=True).fit(X)
    X = scaler.transform(X)

    # Parameter grid declarations.
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=600, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [5,10,20,50,70,100,150,200,300]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    learning_rate = [0.01, 0.05, 0.1]
    min_child_weight = [1, 2, 3, 5, 10]
    subsample = [0.6, 0.7, 0.8, 0.9, 1.0]
    gamma = [0, 0.5, 1, 3, 5, 7, 10]
    colsample_bytree= [0.3, 0.5, 0.7, 1]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'learning_rate':learning_rate,
                   'min_child_weight':min_child_weight,
                   'subsample' : subsample,
                   'colsample_bytree' : colsample_bytree,
                   'gamma' : gamma
                    }

    init_time = time.time()

    xgb = xgb.XGBRegressor(random_state = 42)

    xgb_random = RandomizedSearchCV(estimator=xgb, scoring='neg_mean_squared_error',
                                    param_distributions = random_grid, n_iter = 500,
                                    cv = 3, verbose=2, random_state=42)

    xgb_random.fit(X, y)

    print(xgb_random.best_params_)

    finish_time = time.time()

    print(finish_time-init_time)

    #Save best paramaters and corresponding score in a csv file.

    f = open("xgb_bestparameters.csv", "a")
    f.write(knn_random.best_params_)
    f.write(":")
    f.write(knn_random.best_score_)
    f.write(",")

if __name__ == '__main__':
    main()

