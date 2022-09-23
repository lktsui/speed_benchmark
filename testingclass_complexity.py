import pickle

import tensorflow as tf
import classifier
import load_data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from timeit import default_timer as timer

def get_acc(cf_list):

    accuracy = []

    for cf in cf_list:
        cf_transpose = np.transpose(cf)
        diagonal_sum = np.sum(np.diagonal(cf_transpose))
        total_points = np.sum(cf_transpose)
        accuracy.append(diagonal_sum/total_points)

    return accuracy

def test_train_split(signals, labels):
    test_ratio = 0.2

    train_signals, test_signals, train_labels, test_labels = train_test_split(signals, labels,
                                                                              test_size=test_ratio,
                                                                              stratify=labels)

    dataset_split = {}
    dataset_split['train_signals'] = train_signals
    dataset_split['test_signals'] = test_signals
    dataset_split['train_labels'] = train_labels
    dataset_split['test_labels'] = test_labels

    return dataset_split


def normalization(signals):
    scaler = preprocessing.StandardScaler()
    scaler.fit(signals)
    return scaler

def process_model(model, signals, labels):
    dataset_split = test_train_split(signals, labels)
    scaler = normalization(dataset_split['train_signals'])
    dataset_split['train_signals_normalized'] = scaler.transform(dataset_split['train_signals'])
    dataset_split['test_signals_normalized'] = scaler.transform(dataset_split['test_signals'])

    start = timer()
    model.train(dataset_split['train_signals_normalized'], dataset_split['train_labels'])
    end = timer()

    train_time = end-start

    _, _, inference_time = model.get_acc_metrics(
        dataset_split['train_signals_normalized'],
        dataset_split['train_labels'],
        dataset_split['test_signals_normalized'],
        dataset_split['test_labels'])

    return train_time, inference_time

def benchmark_rf(signals, labels, rf_size, iterations):


    train_time_array = []
    inference_time_array = []

    for iteration in range(0, iterations):
        rf_classifier = classifier.RF_Classifier(rf_size=rf_size)
        train_time, inference_time = process_model(rf_classifier, signals, labels)
        train_time_array.append(train_time)
        inference_time_array.append(inference_time)

    average_train_time = np.average(np.array(train_time_array))
    average_inference_time = np.average(np.array(inference_time_array))

    print("Random Forest: Train time %e 188 pts; inference time %e"%(average_train_time, average_inference_time))

def benchmark_knn(signals, labels, nn_size, iterations):

    train_time_array = []
    inference_time_array = []
    for iteration in range(0, iterations):
        nn_classifier = classifier.KN_Classifier(neighbor_size=nn_size)
        train_time, inference_time = process_model(nn_classifier, signals, labels)
        train_time_array.append(train_time)
        inference_time_array.append(inference_time)

    average_train_time = np.average(np.array(train_time_array))
    average_inference_time = np.average(np.array(inference_time_array))
    print("NN 5: Train time %e 188 pts; inference time %e" % (average_train_time, average_inference_time))

def benchmark_ann(signals, labels, hl_size, iterations):

    train_time_array = []
    inference_time_array = []
    for iteration in range(0, iterations):
        print("Algorithm: ANN Size: %d - Iteration %d" % (hl_size, iteration))
        ann_classifier = classifier.KerasANN_Classifier(3, af='tanh', hl_size=hl_size)
        train_time, inference_time = process_model(ann_classifier, signals, labels)
        train_time_array.append(train_time)
        inference_time_array.append(inference_time)

    average_train_time = np.average(np.array(train_time_array))
    average_inference_time = np.average(np.array(inference_time_array))
    print("ANN 5: Train time %e 188 pts; inference time %e" % (average_train_time, average_inference_time))

def main():
    dataset = load_data.generate_dataset()

    data_type = 'V'
    temperature = 550

    print("Processing Temperature: %s %s\n" % (data_type, str(temperature)))

    signals = pd.concat((dataset['%s0_%d' % (data_type, temperature)],
                         dataset['%s1_%d' % (data_type, temperature)],
                         dataset['%s2_%d' % (data_type, temperature)],
                         ), axis=1)

    labels = np.array(dataset['Label'].values)

    iterations = 10

    # Optimize for Random Forest Size
    rf_size = 10
    benchmark_rf(signals, labels, rf_size, iterations)

    # # Optimize for KNN neighbors
    nn_size = 5
    benchmark_knn(signals, labels, nn_size, iterations)

    # Optimize for ANN size
    hl_size = 10
    benchmark_ann(signals, labels,hl_size, iterations)




if __name__ == '__main__':
    main()
