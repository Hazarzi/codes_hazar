# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 11:52:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 168                                                    $: Revision of the last commit

"""
Drop column feature importances.

This approach is quite an intuitive one, as we investigate the importance of a feature by comparing a model with all
features versus a model with this feature dropped for training.

The feature importances are calculated using out of bag scores.

Pros:
most accurate feature importance
Cons:
potentially high computation cost due to retraining the model for each variant of the dataset (after dropping a single
feature column)
"""
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
    
    #scaler = StandardScaler(copy=True).fit(X)
    
    #X = scaler.transform(X)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
    
    rf = RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)
    rf.fit(X_train, y_train)
    
    
    def dropcol_importances(rf, X_train, y_train):
        """
        Function for calculating oob-scores while dropping a feature.
        Feature-dropped models are then compared to a base model(clone).
        """
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X_train, y_train)
        baseline = rf_.oob_score_
        imp = []
        for col in X_train.columns:
            X = X_train.drop(col, axis=1)
            rf_ = clone(rf)
            rf_.random_state = 999
            rf_.fit(X, y_train)
            o = rf_.oob_score_
            imp.append(baseline - o)
        imp = np.array(imp)
        I = pd.DataFrame(
                data={'Feature':X_train.columns,
                      'Importance':imp})
        I = I.set_index('Feature')
        I = I.sort_values('Importance', ascending=True)
        return I

    
    importance_table = dropcol_importances(rf, X_train, y_train)

    # Transfrom into dataframe
    pd_feat = pd.DataFrame(importance_table)
    pd_feat.index.name = 'feature_name'
    pd_feat.reset_index(inplace=True)
    pd_feat = pd_feat.sort_values(by="Importance", ascending=False)

    # Plot
    var_imp_plot(pd_feat, 'Drop column feature importance \nwith Random Forests')

if __name__ == '__main__':
    main()


"""Second method for drop column features method"""

# from sklearn.ensemble import RandomForestRegressor
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from math import sqrt
# import seaborn as sns
# 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.base import clone
# 
# # function for creating a feature importance dataframe
# def imp_df(column_names, importances):
#     df = pd.DataFrame({'feature': column_names,
#                        'feature_importance': importances}) \
#            .sort_values('feature_importance', ascending = False) \
#            .reset_index(drop = True)
#     return df
# 
# # plotting a feature importance dataframe (horizontal barchart)
# def var_imp_plot(imp_df, title):
#     imp_df.columns = ['Feature', 'Feature Importance']
#     sns.barplot(x = 'Feature Importance', y = 'Feature', data = imp_df, orient = 'h', color = 'royalblue') \
#        .set_title(title, fontsize = 20)
# 
# dataset = pd.read_csv("db_cleaned.csv")
# Y = dataset.TMP
# X = dataset.drop(["TMP"], axis=1)
# 
# #add random feature
# np.random.seed(seed = 42)
# X['RANDOM'] = np.random.random(size = len(X))
# 
# X = pd.get_dummies(X, drop_first=True)
# 
# xcols = X.columns
# 
# # scaling features
# 
# #scaler = StandardScaler(copy=True).fit(X)
# #X = scaler.transform(X)
# 
# X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
# 
# rf = RandomForestRegressor(n_estimators = 100,
#                            n_jobs = -1,
#                            oob_score = True,
#                            bootstrap = True,
#                            random_state = 42)
# rf.fit(X_train, y_train)
# 
# 
# from sklearn.base import clone
# 
# 
# def drop_col_feat_imp(model, X_train, y_train, random_state=42):
#     # clone the model to have the exact same specification as the one initially trained
#     model_clone = clone(model)
#     # set random_state for comparability
#     model_clone.random_state = random_state
#     # training and scoring the benchmark model
#     model_clone.fit(X_train, y_train)
#     benchmark_score = model_clone.score(X_train, y_train)
#     # list for storing feature importances
#     importances = []
# 
#     # iterating over all columns and storing feature importance (difference between benchmark and new model)
#     for col in X_train.columns:
#         model_clone = clone(model)
#         model_clone.random_state = random_state
#         model_clone.fit(X_train.drop(col, axis=1), y_train)
#         drop_col_score = model_clone.score(X_train.drop(col, axis=1), y_train)
#         importances.append(benchmark_score - drop_col_score)
# 
#     importances_df = imp_df(X_train.columns, importances)
#     return importances_df
# 
# drop_imp = drop_col_feat_imp(rf, X_train, y_train)
# var_imp_plot(drop_imp, 'Drop column feature importance \nwith Random Forests')
