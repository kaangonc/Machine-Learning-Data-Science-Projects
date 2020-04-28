import pandas as pn
import numpy as np


def read_files():
    train_features = pn.read_csv("question-3-train-features.csv", header=None, dtype="int")
    train_labels = pn.read_csv("question-3-train-labels.csv", header=None, dtype="int")
    valid_features = pn.read_csv("question-3-valid-features.csv", header=None, dtype="int")
    valid_labels = pn.read_csv("question-3-valid-labels.csv", header=None, dtype="int")
    return pn.DataFrame.to_numpy(train_features, dtype="int"), pn.DataFrame.to_numpy(train_labels, dtype="int"),\
        pn.DataFrame.to_numpy(valid_features, dtype="int"), pn.DataFrame.to_numpy(valid_labels, dtype="int")


def kNN(train_features, train_labels, test_data):
    label_distance_list = []
    index = 0
    for tf_row in train_features:
        distance = -find_cosine_distance(test_data, tf_row)
        label_distance_list.append([train_labels[index], distance])
        index += 1
    sorted_list = sorted(label_distance_list, key=lambda l:l[1])
    return sorted_list

def evaluate(train_features, train_labels, valid_features, valid_labels, k_list):
    true_results = 0
    index = 0
    results = []
    for vf_row in valid_features:
        results.append(kNN(train_features, train_labels, vf_row))
        index += 1

    for k in k_list:
        row_count = 0
        true_results = 0
        positive_predictions = 0
        true_positive = 0
        for row in results:
            count_one = 0
            count_zero = 0
            for j in range(k):
                if row[j][0] == 0:
                    count_zero += 1
                else:
                    count_one += 1

            result_label = 0
            if count_one > count_zero:
                result_label = 1
                positive_predictions += 1

            if valid_labels[row_count] == result_label:
                true_results += 1
                if result_label == 1:
                    true_positive += 1
            row_count += 1
        accuracy = (true_results / len(valid_features)) * 100
        precision = (true_positive / positive_predictions) * 100
        print("Accuracy for k=", k, "is", accuracy, "%")
        print("Precision for k=", k, "is", precision, "%")


def find_cosine_distance(x, y):
    dot_mult = np.dot(x, y)
    magnitude_mult = np.linalg.norm(x) * np.linalg.norm(y)
    return dot_mult / magnitude_mult


train_features, train_labels, valid_features, valid_labels = read_files()
k = [1, 3, 5, 10, 20, 50, 100, 200]
evaluate(train_features, train_labels, valid_features, valid_labels, k)
