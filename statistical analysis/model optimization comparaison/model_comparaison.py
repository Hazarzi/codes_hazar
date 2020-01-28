# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 11:52:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 168                                                    $: Revision of the last commit

"""
Script for model comparaison using 5x2 cross-validated paired t-test method.
Imported modules differ depending on the models to be compared.
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from math import sqrt
import scipy

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def main():
    """
    :return: Returns the p-value of a t-test between two models.
    """
    # Import data from scv file and remove unnecessary subjective variablaes
    df = pd.read_csv("db_kNN_cleaned.csv")

    # Separate features(X)/ dependant(y)
    y = df.TMP
    X = df.drop(["TMP"], axis=1)

    # One hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Scale features
    scaler = StandardScaler(copy=True).fit(X)
    X = scaler.transform(X)

    # Define seed list for each 2 fol iteration.

    seed_list = [582, 634, 256, 974, 724]

    # Models to be compared.
    model1 = RandomForestRegressor()
    model2 = RandomForestRegressor(n_estimators=300,
                                   min_samples_split=3,
                                   min_samples_leaf=1,
                                   max_features="auto",
                                   n_jobs=-1,
                                   oob_score=True,
                                   bootstrap=True)

    # List declarations
    model1_rmse = []
    model2_rmse = []
    iter_mean_score_list = []
    variance_list = []
    diff_sq_list = []

    # Iterate through the seed list for having 5x2 folds.
    for ix, seed in enumerate(seed_list):

        # K-fold declaration with random seed.
        cv = KFold(n_splits=2, random_state=seed)
        iter_score_diff_list = []
        for i_f, (train_index, test_index) in enumerate(cv.split(X)):
            print("Train Index: ", train_index, "\n")
            print("Test Index: ", test_index)
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            model1.fit(X_train, y_train)
            model1_predictions = model1.predict(X_test)
            model1_score = sqrt(mean_squared_error(y_test, model1_predictions))
            model1_rmse.append(model1_score)

            model2.fit(X_train, y_train)
            model2_predictions = model2.predict(X_test)
            model2_score = sqrt(mean_squared_error(y_test, model2_predictions))
            model2_rmse.append(model2_score)

            fold_score_diff = (model1_score - model2_score)
            diff_sq_list.append(fold_score_diff**2)
            iter_score_diff_list.append(fold_score_diff)

        # Calculate mean.
        iter_mean_score = np.mean(iter_score_diff_list)
        iter_mean_score_list.append(iter_mean_score)

        # Calculate variance.
        variance = (iter_score_diff_list[0]-iter_mean_score) ** 2 + (iter_score_diff_list[1]-iter_mean_score) ** 2
        variance_list.append(variance)

        print("Fold %2d score difference = %.6f" % (i_f + 1, model1_score - model2_score))

    # Calculate F-statistic and p-value.
    numerator = sum(diff_sq_list)
    denominator = 2*(sum(variance_list))
    f_stat = numerator / denominator
    p_value = scipy.stats.f.sf(f_stat, 10, 5)
    print("P value is: " + str(p_value))
    return p_value


if __name__ == '__main__':
    main()
