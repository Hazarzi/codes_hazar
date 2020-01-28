Five different scripts for five different methods for calculating feature importances. Multiple methods are presented as
each approach has advantages and disadvantages and a comparaison between methods is more reliable.

For each method, a random variable(which is supposed to have no effect on predictions) is also added to compare dataset features to the random variable.

db_cleaned.csv:
CSV file containing the database for model creation.

random_forests_builtin_importane.py:
Feature importance using Random Forests feature_importance_ method. Based on Gini coefficients.

drop_column_importance.py:
Feature importance using drop column method.The importance of a feature is studied by comparing a model with all
features versus a model with this feature dropped for training.
Feature importances are calculated using out of bag scores of a Random Forests model.

permutation_importance.py:
This approach directly measures feature importance by observing how random re-shuffling (thus preserving the
distribution of the variable) of each predictor influences model performance by using eli5 library.
Feature importances are calculated Random Forests feature_importance_ method.

elastic_net_importance.py:
Multiple models are built by varying the l1 regularization parameter of the Elastic Net model. Regularisation consists
in adding a penalty to the different parameters of the machine learning model to reduce the freedom of the model and
in other words to avoid overfitting.  From the different types of regularisation, Lasso or L1 has the property that is
able to shrink some of the feature coefficients to zero where less important features will be therefore removed from 
the final model.

shap_importance.py:
SHAP(SHapley Additive exPlanations) is a Python library for model explanations and is based on the Shapley Value which 
is a solution concept in cooperative game theory. It can be simplified as the value of contribution of a given feature 
to the model predictions.
Three different tree based models are created and respective Shapley Values are calculated.

