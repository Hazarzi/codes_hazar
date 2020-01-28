# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 20/08/2019

# //$URL::                              $: URL of the file on the svn repository
# //$Author::                           $: Author of the last commit
# //$Date::                             $: Date of the last commit
# //$Revision::                         $: Revision of the last commit

"""
Code for running Tukey's test on 5x2 cross-validated model scores.
"""
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

df = pd.read_csv("5x2results_last.csv")
df.dataset = df.dataset.astype(str)
df.rename(columns={'method':'Algorithm'}, inplace=True)
df.rename(columns={'rmse':'RMSE'}, inplace=True)
df.rename(columns={'dataset':'Dataset'}, inplace=True)

mc = MultiComparison(df['RMSE'], df['Algorithm'])
mc_results = mc.tukeyhsd()
print(mc_results)

sns.set()
sns.set_style("whitegrid")
sns.pointplot(data=df, x='Dataset', y='RMSE',hue="Algorithm", palette='colorblind')

