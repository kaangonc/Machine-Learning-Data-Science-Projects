
import pandas as pn
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier

def read_files():
  train_features = pn.read_csv("q3_train_features.csv", header=None)
  train_binary_labels = pn.read_csv("q3_train_binary_labels.csv", header=None)
  train_multiclass_labels = pn.read_csv("q3_train_multiclass_labels.csv", header=None)
  test_features = pn.read_csv("q3_test_features.csv", header=None)
  test_binary_labels = pn.read_csv("q3_test_binary_labels.csv", header=None)
  test_multiclass_labels = pn.read_csv("q3_test_multiclass_labels.csv", header=None)
  return pn.DataFrame.to_numpy(train_features)[:,1:], pn.DataFrame.to_numpy(train_binary_labels)[:,1:], pn.DataFrame.to_numpy(train_multiclass_labels)[:,1:], pn.DataFrame.to_numpy(test_features)[:,1:], pn.DataFrame.to_numpy(test_binary_labels)[:,1:], pn.DataFrame.to_numpy(test_multiclass_labels)[:,1:]

train_features, train_binary_labels, train_multiclass_labels, test_features, test_binary_labels, test_multiclass_labels = read_files()
# print(train_binary_labels[:,1:])


def divide(nom, dem):
  if dem==0:
    return 0
  return nom / dem
def find_accuracy_and_confusion_matrix(test_values, predictions, class_amount):
  metrics = {}

  # accuracy
  true_predictions = 0
  for j in range(predictions.shape[0]):
    if predictions[j] == test_values[j]:
      true_predictions += 1
  metrics["Accuracy"] = true_predictions / predictions.shape[0]

  # confusion matrix
  metrics["Confusion Matrix"] = np.zeros((class_amount, class_amount))
  for i in range(predictions.shape[0]):
    if test_values[i] == predictions[i]:
      metrics["Confusion Matrix"][test_values[i][0]][test_values[i][0]] += 1
    else:
      metrics["Confusion Matrix"][test_values[i][0]][predictions[i]] += 1
  return metrics


def find_metrics(test_values, predictions, class_value):
  metrics = {}

  #precision
  true_one = 0
  one_prediction = 0
  for j in range(predictions.shape[0]):
    if predictions[j] == class_value:
      one_prediction += 1
      if predictions[j] == test_values[j]:
        true_one += 1
  metrics["Precision"] = divide(true_one,one_prediction)

  #recall
  true_one = 0
  one_sample = 0
  for j in range(predictions.shape[0]):
    if test_values[j] == class_value:
      one_sample += 1
      if predictions[j] == test_values[j]:
        true_one += 1
  metrics["Recall"] = divide(true_one,one_sample)

  #negative predictive value
  true_zero = 0
  zero_prediction = 0
  for j in range(predictions.shape[0]):
    if predictions[j] != class_value:
      zero_prediction += 1
      if predictions[j] == test_values[j]:
        true_zero += 1
  metrics["Negative Predictive Value"] = divide(true_zero,zero_prediction)

  #false positive rate
  false_one = 0
  zero_sample = 0
  for j in range(predictions.shape[0]):
    if test_values[j] != class_value:
      zero_sample += 1
      if predictions[j] == 1:
        false_one += 1
  metrics["False Positive Rate"] = divide(false_one,zero_sample)

  #false discovery rate
  metrics["False Discovery Rate"] = 1 - metrics["Precision"]

  #F1
  metrics["F1"] = divide(2 * metrics["Precision"] * metrics["Recall"] , (metrics["Precision"] + metrics["Recall"]))

  #F2
  metrics["F2"] = divide(5 * metrics["Precision"] * metrics["Recall"], (4 * metrics["Precision"] + metrics["Recall"]))

  return metrics, true_one, true_zero, false_one, (zero_prediction-true_zero)


def find_macro_averages(test_values, predictions, class_amount):
  # #precision
  # precisions = []
  # recalls = []
  # for i in range(confusion_matrix.shape[0]):
  #   true_positive = 0
  #   false_positive = 0
  #   false_negative = 0
  #   for j in range(confusion_matrix.shape[1]):
  #     if i==j:
  #       true_positive += confusion_matrix[i][i]
  #     else:
  #       false_positive += confusion_matrix[j][i]
  #       false_negative += confusion_matrix[i][j]
  #   precisions.append(true_positive/(true_positive+false_positive))

  metrics = {}
  for i in range(class_amount):
    ind_metrics = find_metrics(test_values, predictions, i)[0]
    if i == 0:
      for key in ind_metrics:
        metrics["Macro " + key] = 0
    for key in ind_metrics:
      metrics["Macro " + key] += divide(ind_metrics[key] , class_amount)
  return metrics

def find_micro_averages(test_values, predictions, class_amount):
  metrics = {}
  true_pos_sum = 0
  true_neg_sum = 0
  false_pos_sum = 0
  false_neg_sum = 0
  f1_nominator = 0
  f1_denominator = 0
  f2_nominator = 0
  f2_denominator = 0
  for i in range(class_amount):
    ind_metrics, true_pos, true_neg, false_pos, false_neg = find_metrics(test_values, predictions, i)
    f1_nominator += 2*ind_metrics["Precision"]*ind_metrics["Recall"]
    f2_nominator += 5 * ind_metrics["Precision"] * ind_metrics["Recall"]
    f1_denominator += ind_metrics["Precision"] + ind_metrics["Recall"]
    f2_denominator += 4*ind_metrics["Precision"] + ind_metrics["Recall"]
    true_pos_sum += true_pos
    true_neg_sum += true_neg
    false_pos_sum += false_pos
    false_neg_sum += false_neg

  metrics["Micro Precision"] = divide(true_pos_sum ,(true_pos_sum+false_pos_sum))
  metrics["Micro Recall"] = divide(true_pos_sum , (true_pos_sum + false_neg_sum))
  metrics["Micro Negative Predictive Value"] = divide(true_neg_sum , (true_neg_sum+false_neg_sum))
  metrics["Micro False Positive Rate"] = divide(false_pos_sum , (true_neg_sum + false_pos_sum))
  metrics["Micro False Discovery Rate"] = 1 - metrics["Micro Precision"]
  metrics["Micro F1"] = divide(f1_nominator,f1_denominator)
  metrics["Micro F2"] = divide(f2_nominator , f2_denominator)

  return metrics


def find_optimum(folds, values, parameter, parameter_values, fold_size, k, optimum_c):
  accuracies = np.zeros((k, np.size(c_values)))
  for i in range(k):
    validation_fold = folds[i]
    validation_label = values[i]
    validation_label = np.ravel(validation_label)
    train_folds = np.delete(folds, i, 0)
    train_folds = np.reshape(train_folds, (fold_size * (k - 1), folds.shape[2]))
    train_labels = np.delete(values, i, 0)
    train_labels = np.reshape(train_labels, (fold_size * (k - 1), values.shape[2]))
    train_labels = np.ravel(train_labels)
    count = 0
    for p in parameter_values:
      classifier = None
      if parameter == "c":
        classifier = svm.SVC(C=p, kernel='linear')
      else:
        classifier = svm.SVC(C=optimum_c, gamma=p, kernel = 'rbf')
      classifier.fit(train_folds, train_labels)
      predictions = classifier.predict(validation_fold)
      accuracy = (predictions == validation_label).mean()
      accuracies[i][count] = accuracy
      count += 1

  accuracy_means = np.mean(accuracies, axis=0)
  plt.plot(parameter_values, accuracy_means, '-r', label='')
  if parameter == "c":
    plt.title('C Values vs Accuracy Means')
    plt.xlabel('C Values', color='#1C2833')
  else:
    plt.title('Gamma Values vs Accuracy Means')
    plt.xlabel('Gamma Values', color='#1C2833')
  plt.ylabel('Accuracy Means', color='#1C2833')
  plt.legend(loc='upper left')
  plt.grid()
  plt.show()

  optimum = parameter_values[np.argmax(accuracy_means)]
  return optimum
  # train


def predict(parameter, train_features, train_labels, test_features, optimum_gamma, optimum_c):
  classifier = None
  if parameter == "c":
    classifier = svm.SVC(C=optimum_c, kernel='linear')
  else:
    classifier = svm.SVC(C=optimum_c, gamma=optimum_gamma, kernel='rbf')
  classifier.fit(train_features, np.ravel(train_labels))
  predictions = classifier.predict(test_features)
  return predictions


def prepare(features, labels):
  k = 5
  fold_size = int(features.shape[0] / k)
  folds = []
  values = []
  fold_count = -1
  indices = np.arange(features.shape[0])
  np.random.seed(0)
  indices = np.random.permutation(indices)
  count = 0
  for i in indices:
    if count%fold_size == 0:
      fold_count += 1
      folds.append([])
      values.append([])
    folds[fold_count].append(features[i])
    values[fold_count].append(labels[i])
    count += 1
  folds = np.array(folds)
  values= np.array(values)
  return folds, values, fold_size, k


"""Prepare parts 3.1 and 3.2"""
folds, values,  fold_size, k = prepare(test_features, train_binary_labels)
"""Part 3.1"""
c_values = np.array([10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 100])
optimum_c = find_optimum(folds, values, "c", c_values, fold_size, k, None)
predictions = predict("c", train_features, train_binary_labels, test_features, None, optimum_c)
print("Optimum C Value:", optimum_c)
print(find_accuracy_and_confusion_matrix(test_binary_labels, predictions, 2))
print(find_macro_averages(test_binary_labels, predictions, 2))
print(find_micro_averages(test_binary_labels, predictions, 2))

"""Part 3.2"""
gamma_values = [2**-4, 2**-3, 2**-2, 2**-1, 1, 2]
optimum_gamma = find_optimum(folds, values, "gamma", gamma_values, fold_size, k, optimum_c)
predictions = predict("gamma", train_features, train_binary_labels, test_features, optimum_gamma, optimum_c)
print("Optimum Gamma Value:", optimum_gamma)
print(find_accuracy_and_confusion_matrix(test_binary_labels, predictions, 2))
print(find_macro_averages(test_binary_labels, predictions, 2))
print(find_micro_averages(test_binary_labels, predictions, 2))

"""Part 3.3"""
classifier = OneVsRestClassifier(svm.SVC(C=optimum_c, gamma=optimum_gamma, kernel='rbf'))
classifier.fit(train_features, np.ravel(train_multiclass_labels))
predictions = classifier.predict(test_features)
print(find_accuracy_and_confusion_matrix(test_multiclass_labels, predictions, 3))
print(find_macro_averages(test_binary_labels, predictions, 3))
print(find_micro_averages(test_binary_labels, predictions, 3))