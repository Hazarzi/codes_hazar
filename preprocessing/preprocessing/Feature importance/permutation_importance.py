# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 11:52:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 168                                                    $: Revision of the last commit

""""
Permutation feature importances with Random Forests.

This approach directly measures feature importance by observing how random re-shuffling (thus preserving the
distribution of the variable) of each predictor influences model performance.

The approach can be described in the following steps:
Train the baseline model and record the score (accuracy/R²/any metric of importance) by passing the validation set
(or OOB set in case of Random Forest). This can also be done on the training set, at the cost of sacrificing information
 about generalization.

Re-shuffle values from one feature in the selected dataset, pass the dataset to the model again to obtain predictions
and calculate the metric for this modified dataset. The feature importance is the difference between the benchmark score
and the one from the modified (permuted) dataset.

Repeat 2. for all features in the dataset.

Pros:
applicable to any model
reasonably efficient
reliable technique
no need to retrain the model at each modification of the dataset

Cons:
more computationally expensive than the default feature_importances
permutation importance overestimates the importance of correlated predictors — Strobl et al (2008)
"""
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

def main():
    
    # function for creating a feature importance dataframe
    def imp_df(column_names, importances):
        df = pd.DataFrame({'feature': column_names,
                           'feature_importance': importances}) \
               .sort_values('feature_importance', ascending = False) \
               .reset_index(drop = True)
        return df
    
    # plotting a feature importance dataframe (horizontal barchart)
    def var_imp_plot(imp_df, title):
        imp_df.columns = ['Feature', 'Feature Importance']
        sns.barplot(x = 'Feature Importance', y = 'Feature', data = imp_df, orient = 'h', color = 'royalblue') \
           .set_title(title, fontsize = 20)
    
    dataset = pd.read_csv("db_kNN_cleaned.csv")
    
    Y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)
    
    #add random feature
    np.random.seed(seed = 42)
    X['RANDOM'] = np.random.random(size = len(X))
    
    X = pd.get_dummies(X, drop_first=True)
    
    xcols = X.columns
    
    # scaling features
    
    scaler = StandardScaler(copy=True).fit(X)
    
    X = scaler.transform(X)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
    
    rf = RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)
    rf.fit(X_train, y_train)
    
    
    """eli5: permutation importance"""
    
    perm = PermutationImportance(rf, cv = None, refit = False, n_iter = 50).fit(X_train, y_train)
    perm_imp_eli5 = imp_df(xcols, perm.feature_importances_)
    
    var_imp_plot(perm_imp_eli5, 'Permutation feature importance \nwith Random Forests(eli5)')

    
    """
    Explained as: feature importances
    
    Feature importances, computed as a decrease in score when feature
    values are permuted (i.e. become noise). This is also known as 
    permutation importance.
    
    Feature importances are computed on the same data as used for training, 
    i.e. feature importances don't reflect importance of features for 
    generalization.
    """
    
    print(eli5.format_as_text(eli5.explain_weights(perm)))
    
if __name__ == '__main__':
    main()
