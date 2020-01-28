Two scripts for outlier cleaning.

IQR_clean.py : Removing outliers using 1.5xIQR rule for setting elimination threshold.

kNN_clean.py : Removing outliers/anomalies by fitting a kNN model and eliminating worst performing observations by setting
a threshold of mean+standard deviation using the differences of predicted_y vs true_y of each observation. Observations with predicted values higher than this threshold are eliminated.