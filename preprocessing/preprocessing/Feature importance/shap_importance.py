# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 11:52:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 168                                                    $: Revision of the last commit

"""
Feature Importances using SHAP.

https://www.kaggle.com/wrosinski/shap-feature-importance-with-feature-engineering

SHAP (SHapley Additive exPlanations) is a unified approach to explain the output of any machine learning model.
SHAP connects game theory with local explanations, uniting several previous methods [1-7] and representing the only
possible consistent and locally accurate additive feature attribution method based on expectations (see our papers for
details and citations).

SHAP is based on the Shapley Value which is a solution concept in cooperative game theory.

Three different tree based models are tested.

Great explanation -> https://towardsdatascience.com/interpreting-your-deep-learning-model-by-shap-e69be2b47893
"""

import shap

import pandas as pd #for manipulating data
import numpy as np #for manipulating data
import sklearn #for building models 
import xgboost as xgb #for building models 
import sklearn.ensemble #for building models 
from sklearn.model_selection import train_test_split #for creating a hold-out sample 
import time #some of the routines take a while so we monitor the time 
import os #needed to use Environment Variables in Domino 
import matplotlib.pyplot as plt #for custom graphs at the end 
import seaborn as sns #for custom graphs at the end

def main():
    
    dataset = pd.read_csv("db_kNN_cleaned.csv")

    Y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)
    
    #add random feature
    np.random.seed(seed = 42)
    X['RANDOM'] = np.random.random(size = len(X))
    
    X = pd.get_dummies(X, drop_first=True)

    # Get column names
    xcols = X.columns

    # Split train test.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)

    # XGBoost 
    xgb_model = xgb.train({'objective': 'reg:linear'}, xgb.DMatrix(X_train, label=y_train))
    
    # GBT from scikit-learn 
    sk_xgb = sklearn.ensemble.GradientBoostingRegressor()
    sk_xgb.fit(X_train, y_train)
    
    # Random Forest
    rf = sklearn.ensemble.RandomForestRegressor()
    rf.fit(X_train, y_train)
    
    # K Nearest Neighbor 
    knn = sklearn.neighbors.KNeighborsRegressor()
    knn.fit(X_train, y_train)
    
    # Tree on XGBoost 
    explainerXGB = shap.TreeExplainer(xgb_model)
    shap_values_XGB_test = explainerXGB.shap_values(X_test)
    shap_values_XGB_train = explainerXGB.shap_values(X_train)
    
    # Tree on Scikit GBT 
    explainerSKGBT = shap.TreeExplainer(sk_xgb)
    shap_values_SKGBT_test = explainerSKGBT.shap_values(X_test)
    shap_values_SKGBT_train = explainerSKGBT.shap_values(X_train)
    
    # Tree on Random Forest explainer
    explainerRF = shap.TreeExplainer(rf)
    shap_values_RF_test = explainerRF.shap_values(X_test)
    shap_values_RF_train = explainerRF.shap_values(X_train)
    
    X_train_summary = shap.kmeans(X_train, 10)
    
     # using kmeans 
    t0 = time.time()
    explainerKNN = shap.KernelExplainer(knn.predict, X_train_summary)
    shap_values_KNN_test = explainerKNN.shap_values(X_test)
    shap_values_KNN_train = explainerKNN.shap_values(X_train)
    t1 = time.time()
    timeit = t1 - t0
    timeit
    
    # XGBoost 
    df_shap_XGB_test = pd.DataFrame(shap_values_XGB_test, columns=X_test.columns.values)
    df_shap_XGB_train = pd.DataFrame(shap_values_XGB_train, columns=X_train.columns.values)
    # Scikit GBT 
    df_shap_SKGBT_test = pd.DataFrame(shap_values_SKGBT_test, columns=X_test.columns.values)
    df_shap_SKGBT_train = pd.DataFrame(shap_values_SKGBT_train, columns=X_train.columns.values)
    # Random Forest
    df_shap_RF_test = pd.DataFrame(shap_values_RF_test, columns=X_test.columns.values)
    df_shap_RF_train = pd.DataFrame(shap_values_RF_train, columns=X_train.columns.values)
    # KNN
    df_shap_KNN_test = pd.DataFrame(shap_values_KNN_test, columns=X_test.columns.values)
    df_shap_KNN_train = pd.DataFrame(shap_values_KNN_train, columns=X_train.columns.values)
    
    # if a feature has 10 or less unique values then treat it as categorical
    categorical_features = np.argwhere(np.array([len(set(X_train.values[:, x]))
                                                 for x in range(X_train.values.shape[1])]) <= 10).flatten()

    # Plot feature importances.
    shap.summary_plot(shap_values_XGB_train, X_train, plot_type="bar")

if __name__ == '__main__':
    main()
