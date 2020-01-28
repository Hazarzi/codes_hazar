# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 11:52:10 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 168                                                    $: Revision of the last commit

"""
Feature Importanc using sklearn Random Forests' built in feature importance method.

The condition is based on impurity, which in case of classification problems is Gini impurity/information gain (entropy), while for regression trees its variance. So when training a tree we can compute how much each feature contributes to decreasing the weighted impurity. feature_importances_ in Scikit-Learn is based on that logic, but in the case of Random Forest, we are talking about averaging the decrease in impurity over trees.

Pros:
fast calculation
easy to retrieve — one command
Cons:
biased approach, as it has a tendency to inflate the importance of continuous features or high-cardinality categorical variables
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

def main():
    def var_imp_plot(imp_df, title):
        """
        Function for plotting a feature importance dataframe (horizontal barchart)
        :param imp_df:  Dataframe containing feature names and importance values.
        :param title: Title of the figure.
        :return: Plots a figure.
        """
        imp_df.columns = ['Feature', 'Feature Importance']
        sns.barplot(x = 'Feature Importance', y = 'Feature', data = imp_df, orient = 'h', color = 'royalblue') \
           .set_title(title, fontsize = 20)

    # Load dataset
    dataset = pd.read_csv("db_kNN_cleaned.csv")

    Y = dataset.TMP
    X = dataset.drop(["TMP"], axis=1)

    # Add random feature
    np.random.seed(seed=42)
    X['RANDOM'] = np.random.random(size=len(X))

    X = pd.get_dummies(X, drop_first=True)

    xcols = X.columns

    # Scaling features

    scaler = StandardScaler(copy=True).fit(X)
    X = scaler.transform(X)

    # Split test train
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)

    rf = RandomForestRegressor(n_estimators=100,
                               n_jobs=-1,
                               oob_score=True,
                               bootstrap=True,
                               random_state=42)
    rf.fit(X_train, y_train)
    
    # Print model scores
    print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rf.score(X_train, y_train),                                                                                           rf.oob_score_                                                                                 rf.score(X_valid,
                                                                                                          y_valid)))
    # Calculate feature importances
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    data_line = [xcols[i] for i in indices]

    # Transfrom into dataframe
    pd_data_line = pd.DataFrame(xcols)
    pd_importances = pd.DataFrame(importances)
    pd_data_line['Feature importance'] = pd_importances
    pd_data_line = pd_data_line.sort_values(by="Feature importance", ascending=False)
    pd_data_line.columns = ['Feature', 'Feature Importance']

    # Plot
    var_imp_plot(pd_data_line, "Feature importance with Random Forests' \nbuilt-in feature_importances_ method")

if __name__ == '__main__':
    main()
