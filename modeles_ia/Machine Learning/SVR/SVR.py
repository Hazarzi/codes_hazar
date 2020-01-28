# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 14:41:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 176                                                    $: Revision of the last commit

"""
Script for SVR model creation with tuned hyperparameters. Created model is saved in the current folder.
"""

import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import time

from lib import plot_variables as pv #import script to plot predictions for various feature values
from lib import save_load_model as slm #import script to save/load model

def main():

    # Import dataset

    dataset = pd.read_csv("db_kNN_cleaned.csv")

    # Seperate features(X)/ dependant(y)

    y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)

    # One hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Store feature names 

    names = list(X)

    # Scale features

    scaler = StandardScaler(copy=True).fit(X)
    X = scaler.transform(X)

    # Fitting XGB regressor with parameters obtained from Gridsearch

    scores = []
    mae = []
    mse = []
    rmse = []

    # Prepare model using parameters obtained from GridSearch

    model = SVR(C= 20, epsilon= 0.5, gamma= 'auto', kernel= 'rbf', tol=0.0001)

    # Setup 10 fold Cross-validation(CV)

    cv = KFold(n_splits=10, random_state=42, shuffle=False)

    for train_index, test_index in cv.split(X):
        print("Train Index: ", train_index, "\n")
        print("Test Index: ", test_index)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae.append(mean_absolute_error(y_test, predictions))
        mse.append(mean_squared_error(y_test, predictions))
        rmse.append(sqrt(mean_squared_error(y_test, predictions)))
        scores.append(model.score(X_test, y_test))

    print("Mean MAE:", np.mean(mae))
    print("Mean MSE:", np.mean(mse))
    print("Mean RMSE:", np.mean(rmse))
    print("Mean R-squared:", np.mean(scores))

    # Fit model

    model.fit(X, y)

    method_name = model.__class__.__name__

    # Save model as a pickle file with filename as method name,current time and date

    slm.save_model(method_name,model)

    # Plot model responses in terms various feature values

    pv.plot_variables(scaler, model, method_name, dataset)


if __name__ == '__main__':
    main()
