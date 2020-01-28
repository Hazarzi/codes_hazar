# Copyright (c) 2019 Altran Prototypes Automobiles (APA), Velizy-Villacoublay, France. All rights reserved.
# Author : Hazar Zilelioglu
# Date of creation : 30/07/2019

# //$URL:: https://team.altran.com/svn/APA-AV2A/trunk/WP08_UX/Predictt#$: URL of the file on the svn repository
# //$Author:: hzilelioglu@EUROPE                                       $: Author of the last commit
# //$Date:: 2019-08-07 14:57:22 +0200 (Wed, 07 Aug 2019)               $: Date of the last commit
# //$Revision:: 115                                                    $: Revision of the last commit

"""
Script for creating a simple NN network (2 hidden layers with 128 neurons each) for temperature predictions.
Created model is saved in the current folder. A 5x2 cross validation is done.
"""

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import sqrt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def main():

  # Fix seeds

  tf.reset_default_graph()
  a = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
  graph_level_seed = 1
  operation_level_seed = 1
  tf.set_random_seed(graph_level_seed)
  b = tf.nn.dropout(a, 0.5, seed=operation_level_seed)

  # Import dataset
  
  df = pd.read_csv("db_kNN_cleaned.csv")

  # Separate features(X)/ dependant(y)
  y = df.TMP
  X = df.drop(["TMP"], axis=1)

  # One hot encoding
  X = pd.get_dummies(X, drop_first=True)

  # Scale features
  scaler = StandardScaler(copy=True).fit(X)
  X = scaler.transform(X)


  seed_list = [42]

  print(tf.__version__)

  # Define network architecture.
  
  def build_model():
    model = keras.Sequential([
      layers.Dense(128, activation=tf.nn.tanh , input_shape=[9]),
      layers.Dense(128, activation=tf.nn.tanh),
      layers.Dense(1, activation=tf.nn.relu)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer="sgd",
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

  # Display training progress by printing a single dot for each completed epoch
  class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
      if epoch % 100 == 0: print('')
      print('.', end='')

  EPOCHS = 500

  rmse_list = []

  model = build_model()

  # The patience parameter is the amount of epochs to check for improvement

  early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

  history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                      validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

  plot_history(history)

  loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

  print("Testing set Mean Abs Error: {:5.2f} °C".format(mae))
  print("Testing set Mean Sq Error: {:5.2f} °C".format(mse))

  total_error = tf.reduce_sum(tf.square(tf.sub(y, tf.reduce_mean(y))))
  unexplained_error = tf.reduce_sum(tf.square(tf.sub(y, prediction)))
  R_squared = tf.sub(1, tf.div(unexplained_error, total_error))
  print(rmse_list)

if __name__ == '__main__':
    main()


