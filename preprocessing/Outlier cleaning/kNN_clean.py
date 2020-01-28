# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation :20/08/2019

# //$URL::                                                          $: URL of the file on the svn repository
# //$Author::                                                       $: Author of the last commit
# //$Date::                                                         $: Date of the last commit
# //$Revision::                                                     $: Revision of the last commit

"""
Script for outlier removal using kNN.
"""

import os
import pandas as pd
from sklearn import neighbors


def main():
    
    # Import dataset.
    
    dataset = pd.read_csv("hwtsapcBMI.csv")
    dataset = dataset.drop(["thermalsensationacceptability", "thermalpreference", "thermalcomfort"], axis=1)
    dataset = dataset.sample(frac=1, random_state=231).reset_index(drop=True)
    
    # Separate features(X)/ target(y).
    
    y = dataset.Air_temp
    X = dataset.drop(["Air_temp"], axis=1)
    
    # One hot encoding.
    
    X = pd.get_dummies(X, drop_first=True)
    
    # Define kNN model.
    
    model = neighbors.KNeighborsRegressor(n_neighbors=10)
    
    model.fit(X, y)
    
    # List for storing predictions scores.
    
    mae = []
    
    # Iterate and predict the rows in the dataset.
    
    for index, row in X.iterrows():
        test = pd.DataFrame(row)
        test = test.values.reshape(1, -1)
        pred = model.predict(test)
        mae.append(pred)
    
    # Convert lists to pandas Dataframe for easier manipulation.
    
    mae_df = pd.DataFrame(mae)
    y_true_df = pd.DataFrame(y)
    
    # Calculate the difference between predictions and the ground truth for each row.
    
    diff = y_true_df["Air_temp"] - mae_df[0]
    diff = diff.abs()
    diff = diff.sort_values(ascending=False)
    diff = pd.DataFrame(diff)
    
    # Calculate the mean and standard deviation(std) of the differences . Identify differences with value greater than 
    # mean + std. Remove the observations with differences greater than than mean+std.
    
    diff.mean()
    diff.std()
    threshold = diff.mean() +  diff.std()
    diff_test = diff[diff[0] > threshold[0]]
    diff_test_index = list(diff_test.index)
    cleaned_diff_dataset = dataset.drop(index=diff_test_index).reset_index(drop=True)
    
    # Save cleaned dataset to csv.
    
    path = os.getcwd()
    cleaned_diff_dataset.to_csv(os.path.join(path, r'db_kNN_cleaned.csv'), index=False)


if __name__ == '__main__':
    main()
