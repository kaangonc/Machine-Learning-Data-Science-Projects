import pandas as pn
import numpy as np
from math import e


def read_files():
    train_dataset = pn.read_csv("q4-train-dataset.csv")
    test_dataset = pn.read_csv("q4-test-dataset.csv")
    return pn.DataFrame.to_numpy(train_dataset), pn.DataFrame.to_numpy(test_dataset)


train_dataset, test_dataset = read_files()


# print(train_dataset.shape[1])
# print(np.array([train_dataset[:, -1]]).T)

def generate_initial_coefficients(size):
    w = [np.random.normal(0, 0.01, size)]
    w = np.array(w)
    return w.T


def init_model(dataset):
    features = []
    values = []
    for sample in dataset:
        tmp = [1]
        for j in range(len(sample) - 1):
            tmp.append(float(sample[j]))
        features.append(tmp)
        values.append([float(sample[-1])])

    features = np.array(features)
    # print(features.shape)
    values = np.array(values)
    # print(values.shape)
    normalize(features)
    normalize(values)
    return features, values


def normalize(features):
    min = np.amin(features, axis=0)
    max = np.amax(features, axis=0)
    for i in range(features.shape[0]):
        for j in range(features.shape[1] - 1):
            features[i][j + 1] = (features[i][j + 1] - min[j + 1]) / (max[j + 1] - min[j + 1])


def prob_one(features, w):
    # return 1 / (1+math.exp(np.dot(features,w)))
    # print(w.shape)
    # print(np.dot(features,w))
    f = np.dot(features, w)[0][0]
    ef = np.exp(-f)
    # print(ef)
    # print(1/(1+ef))
    return 1 / (1 + ef)


def find_gradient(x, y, w):
    gradient = []
    # print(x.shape)
    for i in range(x.shape[1]):  # 30
        gradient_i = 0
        for j in range(x.shape[0]):  # 800
            # print(x[j][i])
            gradient_i += x[j][i] * (y[j][0] - prob_one(np.array([x[j]]), w))
        # print(gradient_i)
        gradient.append([gradient_i])

    # print("Gradient shape:", np.array(gradient).shape)
    return np.array(gradient)


def full_batch_gradient_asc(x, y, learning_rate, iteration_amount):
    w = generate_initial_coefficients(x.shape[1])
    # print("w0:", w)
    for i in range(iteration_amount):
        # print("STEP")
        gradient = find_gradient(x, y, w)
        # print("gs: ", gradient.shape)

        # w, gradient: matrix
        w = w + np.dot(learning_rate, gradient)
        # if i == 0:
        #   print("w:", w)
        #   print("gradient:", gradient)
    print(w)
    return w


def stochastic_gradient_asc(x, y, learning_rate, iteration_amount):
    w = generate_initial_coefficients(x.shape[1])
    for i in range(iteration_amount):
        for j in range(x.shape[0]):
            instance = np.array([x[j]])
            instance_value = np.array([y[j]])
            # print(instance.shape)
            gradient = find_gradient(instance, instance_value, w)
            w = w + np.dot(learning_rate, gradient)
    return w


def mini_batch_gradient_asc(x, y, learning_rate, iteration_amount, batch_size):
    w = generate_initial_coefficients(x.shape[1])
    batches = {}
    values = {}
    batch_count = -1
    for i in range(x.shape[0]):
        if i % batch_size == 0:
            batch_count += 1
            batches[batch_count] = []
            values[batch_count] = []
        batches[batch_count].append(x[i])
        values[batch_count].append(y[i])
    for batch in batches:
        batches[batch] = np.array(batches[batch])
        values[batch] = np.array(values[batch])
    print(batches[0].shape)
    print(values[0].shape)
    for i in range(iteration_amount):
        for batch in batches:
            gradient = find_gradient(batches[batch], values[batch], w)
            w = w + np.dot(learning_rate, gradient)
    return w


def predict(test_features, w):
    print("Entered")
    predictions = []
    for j in range(test_features.shape[0]):
        if prob_one(np.array([test_features[j]]), w) > 0.5:
            predictions.append([1])
        else:
            predictions.append([0])
    # print(np.array(predictions))
    return np.array(predictions)


def find_metrics(test_values, predictions):
    metrics = {}

    # accuracy
    true_predictions = 0
    for j in range(predictions.shape[0]):
        if predictions[j][0] == test_values[j][0]:
            true_predictions += 1
    metrics["Accuracy"] = true_predictions / predictions.shape[0]

    # precision
    true_one = 0
    one_prediction = 0
    for j in range(predictions.shape[0]):
        if predictions[j][0] == 1:
            one_prediction += 1
            if predictions[j][0] == test_values[j][0]:
                true_one += 1
    metrics["Precision"] = true_one / one_prediction

    # recall
    true_one = 0
    one_sample = 0
    for j in range(predictions.shape[0]):
        if test_values[j][0] == 1:
            one_sample += 1
            if predictions[j][0] == test_values[j][0]:
                true_one += 1
    metrics["Recall"] = true_one / one_sample

    # negative predictive value
    true_zero = 0
    zero_prediction = 0
    for j in range(predictions.shape[0]):
        if predictions[j][0] == 0:
            zero_prediction += 1
            if predictions[j][0] == test_values[j][0]:
                true_zero += 1
    metrics["Negative Predictive Value"] = true_zero / zero_prediction

    # false positive rate
    false_one = 0
    zero_sample = 0
    for j in range(predictions.shape[0]):
        if test_values[j][0] == 0:
            zero_sample += 1
            if predictions[j][0] == 1:
                false_one += 1
    metrics["False Positive Rate"] = false_one / zero_sample

    # false discovery rate
    metrics["False Discovery Rate"] = 1 - metrics["Precision"]

    # F1
    metrics["F1"] = 2 * metrics["Precision"] * metrics["Recall"] / (metrics["Precision"] + metrics["Recall"])

    # F2
    metrics["F2"] = 5 * metrics["Precision"] * metrics["Recall"] / (4 * metrics["Precision"] + metrics["Recall"])

    # confusion matrix
    metrics["Confusion Matrix"] = np.array([[true_one, false_one], [true_zero, (zero_prediction - true_zero)]])

    return metrics


def perform(train_dataset, test_dataset):
    train_features, train_values = init_model(train_dataset)
    test_features, test_values = init_model(test_dataset)

    w = full_batch_gradient_asc(train_features, train_values, 0.001, 1000)
    predictions = predict(test_features, w)
    metrics = find_metrics(test_values, predictions)


# perform(train_dataset, test_dataset)

# create_model(train_dataset, test_dataset)

train_features, train_values = init_model(train_dataset)
test_features, test_values = init_model(test_dataset)

# def find_learning_rate():
# learning_rate = 0.00001
# for i in range(5):
#   print("Learning Rate", learning_rate, ":")
#   w = full_batch_gradient_asc(train_features, train_values, 0.1, 1000)
#   predictions = predict(test_features, w)
#   metrics = find_metrics(test_values, predictions)
#   print(metrics)
#   learning_rate = learning_rate * 10
# find_learning_rate()

# w = full_batch_gradient_asc(train_features, train_values, 0.1, 1000)
# predictions = predict(test_features, w)
# metrics = find_metrics(test_values, predictions)
# print(metrics)

w = stochastic_gradient_asc(train_features, train_values, 0.1, 1000)
predictions = predict(test_features, w)
# print(predictions)
metrics = find_metrics(test_values, predictions)
print(metrics)
#
# w = mini_batch_gradient_asc(train_features, train_values, 0.1, 1000, 32)
# predictions = predict(test_features, w)
# # print(predictions)
# metrics = find_metrics(test_values, predictions)
# print(metrics)
