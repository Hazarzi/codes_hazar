# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 11:52:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 168                                                    $: Revision of the last commit

"""
Feature importances using Elastic Net.

Multiple models are built by varying the l1 regularization parameter of the Elastic Net model. Regularisation consists
in adding a penalty to the different parameters of the machine learning model to reduce the freedom of the model and
in other words to avoid overfitting.  From the different types of regularisation, Lasso or L1 has the property that is
able to shrink some of the feature coefficients to zero where less important features will be therefore removed from 
the final model.
"""
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

def main():
        
    dataset = pd.read_csv("db_kNN_cleaned.csv")

    Y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)
    
    # Add random feature
    np.random.seed(seed = 42)
    X['random'] = np.random.random(size = len(X))
    
    X = pd.get_dummies(X, drop_first=True)
    
    xcols = X.columns
    
    # scaling features
    
    scaler = StandardScaler(copy=True).fit(X)
    
    X = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
    
    
    from sklearn.linear_model import ElasticNetCV, ElasticNet
    
    cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True,
                            normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5,
                            copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
    
    cv_model.fit(X_train, y_train)
    
    print('Optimal alpha: %.8f'%cv_model.alpha_)
    print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
    print('Number of iterations %d'%cv_model.n_iter_)
    
    model = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)
    model.fit(X_train, y_train)
    
    print(r2_score(y_train, model.predict(X_train)))
    
    feature_importance = pd.Series(index = xcols, data = np.abs(model.coef_))
    
    n_selected_features = (feature_importance>0).sum()
    print('{0:d} features, reduction of {1:2.2f}%'.format(
        n_selected_features,(1-n_selected_features/len(feature_importance))*100))
    
    def var_imp_plot(imp_df, title):
        imp_df.columns = ['Feature', 'Feature Importance']
        sns.barplot(x = 'Feature Importance', y = 'Feature', data = imp_df, orient = 'h', color = 'royalblue') \
           .set_title(title, fontsize = 20)
    
    
    
    pd_feat = pd.DataFrame(feature_importance)
    pd_feat.index.name = 'feature_name'
    pd_feat.reset_index(inplace=True)
    pd_feat = pd_feat.sort_values(by=0, ascending=False)
    
    var_imp_plot(pd_feat, "Feature importance with Elastic Net")
    
if __name__ == '__main__':
    main()
