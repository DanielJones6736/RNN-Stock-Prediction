# -*- coding: utf-8 -*-
"""
Daniel Jones
Stock Market Analysis (of Apple Stock)
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def main():
  # constants
  train_ratio = 0.8

  # importing dataset
  df = pd.read_csv("aapl.us.txt")
  df.head()

  # splitting dataset into test and train dataset
  split_size = int(len(df) * train_ratio)
  train = df[0:split_size]
  test = df[split_size:len(df)]

  train = train.loc[:, ["Open"]].values

  # rescale values to 0,1
  scalar = MinMaxScaler(feature_range=(0, 1))
  train_scaled = scalar.fit_transform(train)

  end_len = len(train_scaled)
  X_train = []
  y_train = []
  timesteps = 40

  # add in 40 instances of sequenced data
  for i in range(timesteps, end_len):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
  X_train, y_train = np.array(X_train), np.array(y_train)

  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  print("X_train --> ", X_train.shape)
  print("y_train shape --> ", y_train.shape)

  # init model
  layers = [X_train.shape[1], 50, 50, 1]
  RNN_model = RNN(layers, "tanh", 0.1)

  # train model
  loss = 0
  for x in X_train:
    result, _ = RNN_model.feed_forward(x.T)

    # MSE loss
    target = x[0][0]
    net = result - target
    loss += ((net) ** 2) / len(X_train)

    RNN_model.backpropagate(net)

  print("Loss: " + str(loss[0][0]))

  real_price = test.loc[:, ["Open"]].values
  print("Real Price Shape --> ", real_price.shape)

  dataset_total = pd.concat((df["Open"], test["Open"]), axis=0)
  inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1, 1)
  inputs = scalar.transform(inputs)

  # reformat data for predictions
  X_test = []

  for i in range(timesteps, real_price.shape[0] + 40):
    X_test.append(inputs[i - timesteps:i, 0])
  X_test = np.array(X_test)

  print("X_test shape --> ", X_test.shape)

  # predict
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  predictions = []
  loss = 0
  for x in X_test:
    result, _ = RNN_model.feed_forward(x.T)
    predictions.append(result[0][0])

    # MSE for now
    target = x[0][0]
    net = result[0][0] - target
    loss += ((net) ** 2) / len(X_train)

  print("Loss: " + str(loss))

  fit_pred = np.reshape(predictions, (len(predictions), 1))
  fit_pred = scalar.inverse_transform(fit_pred)

  plt.plot(real_price, color="red", label="Real Stock Price")
  plt.plot(fit_pred, color="black", label="Predicted Stock Price")
  plt.title("Stock Price Prediction")
  plt.xlabel("Time")
  plt.ylabel("Apple Stock Price")
  plt.legend()
  plt.show()


# RNN class that has 2 hidden layers of equal size
class RNN:
  def __init__(self, layers, activ, lr):
    self.activ = activ
    self.lr = lr
    # init weights
    self.weights_ih = np.random.randn(layers[0], layers[1])
    self.weights_hh = np.random.randn(layers[1], layers[2])
    self.weights_ho = np.random.randn(layers[2], layers[3])

    # init biases as 0
    self.bias_h = np.zeros((layers[1], 1))
    self.bias_o = np.zeros((layers[3], 1))

  # activation for tanh
  def activate(self, value, sigma):
    if self.activ == "tanh":
      return np.tanh(self.weights_ih.T @ value + self.weights_hh @ sigma + self.bias_h)
    else:
      return 0

  # feed through network
  def feed_forward(self, input):
    sigma = np.zeros((self.weights_hh.shape[0],1))


    self.prev = input
    self.prev_sigma = [sigma]

    # push through hidden layers
    for i, value in enumerate(input):
      value = np.reshape(value, (value.shape[0], 1))
      sigma = self.activate(value, sigma)
      self.prev_sigma.append(sigma)



    # final output value
    output = self.weights_ho.T @ sigma + self.bias_o
    return output, sigma

  # backpropagate
  def backpropagate(self, del_o):
    num = len(self.prev_sigma)


    # first step of backprop for output layer
    del_bias_o = del_o
    del_weights_ho = del_o @ self.prev_sigma[num-1].T
    del_h = self.weights_ho @ del_o


    # initialize weight change lists to empty
    del_bias_h = np.zeros(self.bias_h.shape)
    del_weights_hh = np.zeros(self.weights_hh.shape)
    del_weights_ih = np.zeros(self.weights_ih.shape)

    # Backpropagation backwards in the sequence
    for timestep in reversed(range(num)):
      inter = (del_h * (1-self.prev_sigma[timestep]**2))
      del_bias_h += inter
      del_weights_hh += inter @ self.prev_sigma[timestep].T

      old_val = np.reshape(self.prev[timestep-1], (self.prev[timestep-1].shape[0], 1))
      del_weights_ih += old_val @ inter.T
      # calc new dL/dh
      del_h = self.weights_hh @ inter

    # update values
    self.weights_ih = np.tanh(self.weights_ih - self.lr * del_weights_ih)
    self.weights_hh = np.tanh(self.weights_hh - self.lr * del_weights_hh)
    self.weights_ho = np.tanh(self.weights_ho - self.lr * del_weights_ho.T)
    self.bias_h = np.tanh(self.bias_h - self.lr * del_bias_h)
    self.bias_o = np.tanh(self.bias_o - self.lr * del_bias_o)


if __name__ == "main":
  main()

