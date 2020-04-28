
import pandas as pn
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def read_file():
  train_features = pn.read_csv("q2-train-features.csv")
  return pn.DataFrame.to_numpy(train_features)

def train(train_features, train_values):
  coefficients = np.dot(train_features.T, train_features)
  coefficients = inv(coefficients)
  coefficients =  np.dot(coefficients, train_features.T)
  coefficients =  np.dot(coefficients, train_values)
  return coefficients


def create_model_A(train_features):
  features = []
  values = []
  for i in train_features:
    features.append([1, int(i[1])])
    values.append([float(i[2])])
  features = np.array(features)
  values = np.array(values)
  return features, values

def create_model_B(train_features):
  features = []
  values = []
  for i in train_features:
    features.append([1, int(i[1]), float(i[3])])
    values.append([float(i[2])])
  features = np.array(features)
  values = np.array(values)
  return features, values


def create_model_C(train_features):
  features = []
  values = []
  for i in train_features:
    features.append([1, int(i[1]), float(i[3]), float(i[4])])
    values.append([float(i[2])])
  features = np.array(features)
  values = np.array(values)
  return features, values


def perform_2_2(train_features):
  features, values = create_model_A(train_features)

  coefficients = train(features, values)
  print(coefficients)

  predictions = find_predictions(coefficients, features)
  
  plott(features, values, predictions)

  print(find_mse(values, predictions))


def perform_2_3(train_features):
  features, values = create_model_B(train_features)

  coefficients = train(features, values)
  print(coefficients)

  predictions = find_predictions(coefficients, features)

  plott(features, values, predictions)

  print(find_mse(values, predictions))


def perform_2_4(train_features):
  features, values = create_model_C(train_features)

  coefficients = train(features, values)
  print(coefficients)

  predictions = find_predictions(coefficients, features)

  plott(features, values, predictions)

  print(find_mse(values, predictions))


def perform_2_5(training_set, test_set):
  features, values = create_model_A(training_set)
  coefficients = train(features, values)
  test_features, test_values = create_model_A(test_set)
  predictions = find_predictions(coefficients, test_features)
  mse_a = find_mse(test_values, predictions)

  features, values = create_model_B(training_set)
  coefficients = train(features, values)
  test_features, test_values = create_model_B(test_set)
  predictions = find_predictions(coefficients, test_features)
  mse_b = find_mse(test_values, predictions)

  features, values = create_model_C(training_set)
  coefficients = train(features, values)
  test_features, test_values = create_model_C(test_set)
  predictions = find_predictions(coefficients, test_features)
  mse_c = find_mse(test_values, predictions)

  print("MSE of model A:", mse_a)
  print("MSE of model B:", mse_b)
  print("MSE of model C:", mse_c)


def plott(features, values, predictions):
  x = features.T
  plt.plot(x[1], predictions, '-r', label='Predictions Curve')
  plt.plot(values, 'ro', label='Ground Values')
  plt.title('USD/TRY exchange rate vs. "Months Past Since November 2014"')
  plt.xlabel('Months Past Since November 2014', color='#1C2833')
  plt.ylabel('USD/TRY exchange rate', color='#1C2833')
  plt.legend(loc='upper left')
  plt.grid()
  plt.show()


def find_predictions(coefficients, features):
  predictions = 0
  x = features.T
  for i in range(coefficients.shape[0]):
    predictions += coefficients[i][0] * x[i]
  return predictions


def find_mse(values, predictions):
  sum = 0
  for i in range(len(predictions)):
    dif = values[i][0] - predictions[i]
    sum += dif**2

  mse = sum / (i+1)
  return mse


train_features = read_file()
perform_2_2(train_features)
perform_2_3(train_features)
perform_2_4(train_features)
perform_2_5(train_features[:-3], train_features[-3:])