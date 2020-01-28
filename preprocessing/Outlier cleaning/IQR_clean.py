# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-20 14:17:21 +0200 (Tue, 20 Aug 2019)               $: Date of the last commit
# //$Revision:: 173                                                    $: Revision of the last commit

"""
Script for outlier removal using 1.5xIQR rule.
"""

import pandas as pd
import os

def main():

    df = pd.read_csv("hwtsapcBMI.csv")

    # Drop subjective features.

    subj = df[["thermalsensationacceptability","thermalpreference","thermalcomfort","thermalsensation"]]
    df = df.drop(["thermalsensationacceptability","thermalpreference","thermalcomfort","thermalsensation"], axis=1)

    # Drop non numerical variables.

    excluded_df = df.select_dtypes(exclude='number')
    num_df = df.select_dtypes(include='number')

    # Define quartiles and IQR.

    Q1 = num_df.quantile(0.25)
    Q3 = num_df.quantile(0.75)
    IQR = Q3 - Q1


    # Remove extreme values based on IQR rule.

    num_df = num_df[~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)]


    # Stitch back dropped columns.

    num_df_index = list(num_df.index)
    num_df = num_df.join(excluded_df.iloc[num_df_index])
    num_df = num_df.join(subj.iloc[num_df_index])

    # Save cleaned file.

    path = os.getcwd()
    num_df.to_csv(os.path.join(path,r'db_IQR_cleaned.csv'), index=False)

if __name__ == "__main__":
    main()





