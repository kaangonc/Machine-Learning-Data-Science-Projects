import csv

import pandas as pn
import numpy as np
import math

V = 37358
LABEL_AMOUNT = 4
ALPHA = 1
IS_DIRICHLET = True

def read_files():
    t_list = pn.read_csv("t-list.csv", header=None, dtype="int32")
    amount = pn.read_csv("vocab-amount.csv", header=None, dtype="int32")
    priors = pn.read_csv("priors.csv", header=None, dtype="int32")
    return pn.DataFrame.to_numpy(t_list, dtype="int32"), pn.DataFrame.to_numpy(amount, dtype="int32"),\
        pn.DataFrame.to_numpy(priors, dtype="int32")


def train():
    train_labels = pn.DataFrame.to_numpy(pn.read_csv("question-4-train-labels.csv", header=None, dtype="int"), dtype="int")

    vocab_list = []
    f = open("question-4-vocab.txt", "r")
    for x in f:
        vocab_list.append([x])

    vocab_amount_list_tmp = np.zeros((LABEL_AMOUNT, V, 2))
    for l in range(LABEL_AMOUNT):
        for v in range(V):
            vocab_amount_list_tmp[l][v] = [v, 0]

    vocab_amount_list = np.array(vocab_amount_list_tmp)

    label_amount_list = [0] * 4
    count = 0
    for i in train_labels:
        index = int(train_labels[i][0])
        label_amount_list[index] += 1
        count += 1
    prior_list = np.array([x / count for x in label_amount_list])

    t_list = np.zeros((LABEL_AMOUNT, V))
    total_words_per_label = np.zeros(LABEL_AMOUNT)
    with open("question-4-train-features.csv") as f:
        reader = csv.reader(f)
        row_count = 0
        for row in reader:
            col_count = 0
            for j in row:
                index = int(train_labels[row_count][0])
                t_list[index][col_count] += int(j)
                total_words_per_label[index] += int(j)
                vocab_amount_list[index][col_count][1] += int(j)
                col_count += 1
            row_count += 1

    sorted_vocab_amount_list = []

    for j in range(LABEL_AMOUNT):
        sorted_vocab_amount_list.append(sorted(vocab_amount_list[j], key=lambda l:l[1], reverse=True))

    for l in range(LABEL_AMOUNT):
        for i in range(20):
            print(vocab_list[int(sorted_vocab_amount_list[l][i][0])])



    return t_list, total_words_per_label, prior_list, sorted_vocab_amount_list

def generateThetaList(t_list, total_words_per_label):
    theta_label_list = []
    t_count = 0
    for t_row in t_list:
        theta_label_list.append([float(x / total_words_per_label[t_count]) for x in t_row])
        t_count += 1

    return  theta_label_list

def generateThetaListWithSmoothing(t_list, total_words_per_label):
    theta_label_list = []
    t_count = 0
    for t_row in t_list:
        theta_label_list.append([float((x + ALPHA) / (total_words_per_label[t_count] + (V * ALPHA))) for x in t_row])
        t_count += 1

    return theta_label_list

def test(t_list, total_words_per_label, prior_list):
    test_labels = pn.DataFrame.to_numpy(pn.read_csv("question-4-test-labels.csv", header=None, dtype="int"), dtype="int")

    if IS_DIRICHLET:
        theta_label_list = generateThetaListWithSmoothing(t_list, total_words_per_label)
    else:
        theta_label_list = generateThetaList(t_list, total_words_per_label)

    with open("question-4-test-features.csv") as f:
        maps_for_labels = np.zeros((4, 800))
        confusion_matrix = np.zeros((LABEL_AMOUNT, LABEL_AMOUNT))
        reader = csv.reader(f)
        row_count = 0
        true_results = 0
        for row in reader:
            map_list = []
            for k in range(LABEL_AMOUNT):
                map = calculate_map(theta_label_list[k], row, prior_list[k])
                map_list.append(map)
                maps_for_labels[k][row_count] = map
            prediction = predict(map_list)
            label = test_labels[row_count][0]
            confusion_matrix[label][prediction] += 1
            if prediction == label:
                true_results += 1
            row_count += 1

        print("Maps For Labels:")
        print(maps_for_labels)
        print("Max-Min")
        for i in maps_for_labels:
            a = np.argsort(i)
            b = np.sort(i)
            print("Max:", a[-1], b[-1])
            print("Min:", a[0], b[0])
        print("Confusion Matrix")
        print(confusion_matrix)
        print("True Results:", true_results)
        print("False Results:", row_count-true_results)
        accuracy = (true_results / row_count) * 100
        return accuracy


def calculate_map(theta_list, document, prior):
    result = c_log_n(1, prior)
    for j in range(V):
        result += c_log_n(int(document[j]), theta_list[j])
    return result


def predict(map_list):
    index = 0
    max_map = map_list[0]
    for k in range(len(map_list) - 1):
        if map_list[k + 1] > max_map:
            max_map = map_list[k + 1]
            index = k + 1
    if map_list.count(max_map) > 1:
        # In case of tie, predict Space
        return 1
    return index

def find_max_min(list):
    max_min_map = np.zeros((4, 4))
    for i in range(LABEL_AMOUNT):
        max_index = 0
        min_index = 0
        max_map = list[i][0]
        min_map = list[i][0]
        for k in range(799):
            if list[i][k + 1] > max_map:
                max_map = list[i][k + 1]
                max_index = k + 1
            if list[i][k + 1] < min_map:
                min_map = list[i][k + 1]
                min_index = k + 1
        max_min_map[i] = [max_index, max_map, min_index, min_map]
    return max_min_map

def find_min(list):
    index = 0
    min_map = list[0]
    for k in range(len(list) - 1):
        if list[k + 1] < min_map:
            min_map = list[k + 1]
            index = k + 1
    return index, min_map


def c_log_n(c, n):
    if c == 0 and n == 0:
        return 0
    if n == 0:
        return -math.inf
    return c * np.math.log(n)


t_list, total_words_per_label, prior_list, vocab = train()
accuracy = test(t_list, total_words_per_label, prior_list)
print("Final Accuracy:", accuracy)