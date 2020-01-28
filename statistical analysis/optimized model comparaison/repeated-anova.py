# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 20/08/2019

# //$URL::                              $: URL of the file on the svn repository
# //$Author::                           $: Author of the last commit
# //$Date::                             $: Date of the last commit
# //$Revision::                         $: Revision of the last commit

"""
Code for running repeated-measures on 5x2 cross-validated model scores.
"""

import pandas as pd
import scipy.stats as stats
import researchpy as rp
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import pingouin
import seaborn as sns

df = pd.read_csv("5x2results_last.csv")
df.dataset = df.dataset.astype(str)
df.rename(columns={'method':'Algorithm'}, inplace=True)
df.rename(columns={'rmse':'RMSE'}, inplace=True)
df.rename(columns={'dataset':'Fold'}, inplace=True)

# Dataset summary

rp.summary_cont(df['RMSE'].groupby(df['Algorithm']))

"""
Sphericity, normality and homogeneity are verified before proceeding to
repeated-measures Anova. Sphericity results us obtained after the rmAnova.
"""

# Levene's test for homogeneity.

levene = stats.levene(df['RMSE'][df['Algorithm'] == 'rf'],
             df['RMSE'][df['Algorithm'] == 'xgb'],
             df['RMSE'][df['Algorithm'] == 'knn'],
             df['RMSE'][df['Algorithm'] == 'nn'],
             df['RMSE'][df['Algorithm'] == 'svr'],
             df['RMSE'][df['Algorithm'] == 'mlp'],
             df['RMSE'][df['Algorithm'] == 'lr'],
             )

print(levene)

# Test of normality. Data is fitted to an ordinary least squares model first.

results = ols('RMSE ~ C(Algorithm)', data=df).fit()

print(stats.shapiro(results.resid))

print(pingouin.rm_anova(data=df, dv="RMSE", within="Algorithm", subject="Dataset"))

sns.set()
sns.set_style("whitegrid")
sns.pointplot(data=df, x='Dataset', y='RMSE',hue="Algorithm", palette='colorblind')
