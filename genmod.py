#!/usr/bin/env python3
#
# Probabilistic generative model for predicting vehicle classes
#
# @author David Rubin
import vehicle
import plotit
import os
import csv
import math
import numpy as np


def class_prior(data, classNo):
    """
    Maximum likelihood for the prior probability (Gaussian dist.)

    :param data: vehicle dictionary with classes
    :param classNo: the class for which we are calculating the probability
    :return: probability of the class in the data
    """
    n_p = np.shape(np.nonzero(data['C'] == classNo))[1]
    n = np.shape(data['C'])[0]
    return n_p / n


def mean_mle(data, classNo, features):
    """
    Calculate the MLE of the mean for a given feature vector of class classNo

    :param data: the vehicle data dict
    :param classNo: the ID of the class we want to calculate the mean for
    :param features: how many features (from 0) should we use (2 ... data.shape[1])
    :return: vector of means for each feature
    """
    features = vehicle.extract_features(data, classNo, features)
    feature_sum = np.sum(features, axis=0)
    return feature_sum / len(features)


def shared_covariance(data, class1, class2, features):
    """
    Compute the MLE of the shared covariance matrix

    :param data: the vehicle data (only for class1 and class2)
    :param class1: the first class ID
    :param class2: the second class ID
    :param features: the number of features used (should not be higher than 2nd dimension of data)
    :return: shared covariance matrix (see Appendix on NN19_task2.pdf)
    """
    # Calculate the MLE means
    mean1 = mean_mle(data, class1, features)
    mean2 = mean_mle(data, class2, features)
    # Calculate the S matrices
    s1 = s_i(data, class1, mean1, features)
    s2 = s_i(data, class2, mean2, features)
    # Get the prior probability for each class
    p1 = class_prior(data, class1)
    p2 = class_prior(data, class2)

    # Return the weighted sum of the matrices for each class
    return np.add(p1 * s1, p2 * s2)


def s_i(data, classNo, mi, features):
    """
    Compute the S_i from the slides (see Appendix on NN19_task2.pdf)

    :param data: vehicle data
    :param classNo: the class (i) ID
    :param mi: the MLE of the mean
    :param features: how many features we use from the data
    :return: matrix S_i
    """
    # Extract features for given class only
    features = vehicle.extract_features(data, classNo, features)

    feature_sum = np.zeros((features[0].shape[0], features[0].shape[0]))
    for feature in features:
        norm_feat = np.array([feature - mi])
        feature_sum = np.add(feature_sum, np.dot(norm_feat.T, norm_feat))
    return np.true_divide(feature_sum, features.shape[0])


def sigmoid(x):
    """The logistic sigmoid function"""
    # return np.exp(x) / (1 + np.exp(x))
    return 1 / (1 + math.e ** -x)


def weight0(eps, mi1, mi2, p1, p2):
    """
    Calculate the w0 from the posterior distribution
    :param eps: covariance matrix
    :param mi1: mle mean feature vector for class 1
    :param mi2: mle mean feature vector for class 2
    :param p1: prior probability of class 1
    :param p2: prior probability of class 2
    :return: float representing w0
    """
    covar_inv = np.linalg.inv(eps)
    # Note: mi1 and mi2 are row vectors, so the transpose operations are inverted
    return (-1/2 * np.dot(np.dot(mi1, covar_inv), mi1.T)
            + 1/2 * np.dot(np.dot(mi2, covar_inv), mi2.T) + math.log(p1/p2))[0][0]


def weight(eps, mi1, mi2):
    """
    Calculate the weight vector using the covariance matrix and the means
    :param eps: the covariance matrix
    :param mi1: mle mean for feature vector of class 1
    :param mi2: mle mean for feature vector of class 2
    :return: the weight vector (a column vector!)
    """
    covar_inv = np.linalg.inv(eps)
    return np.dot(covar_inv, (mi1 - mi2).T)


def predict(sample, covar, mi1, mi2, p1, p2):
    """
    Predict if the sample belongs to class1 (=0) or class2 (=1)
    :param sample: the feature set of the sample we are classifying
    :param covar: the covariance matrix
    :param mi1: mean feature vector of class1
    :param mi2: mean feature vector of class2
    :param p1: prior probability of class1
    :param p2: prior probability of class2
    :return: 1 if class1 and 0 if class2
    """
    return sigmoid(np.dot(weight(covar, mi1, mi2).T, sample) + weight0(covar, mi1, mi2, p1, p2))


def train_model(data, features, class1=1, class2=0):
    """
    Calculate the required arguments for the prediction function

    covar   ... covariance matrix for the selected number of features
    mi1     ... the MLE of the mean for class 1
    mi2     ... the MLE of the mean for class 2
    prior1  ... the prior of class 1
    prior2  ... the prior of class 2

    :param data: the training data
    :param class1: class 1 ID
    :param class2: class 2 ID
    :param features: the number of features to use
    :return: covariance, mi1, mi2, prior1, prior2
    """
    covar = shared_covariance(data, class1, class2, features)
    mi_class1 = np.array([mean_mle(data, class1, features)])
    mi_class2 = np.array([mean_mle(data, class2, features)])
    prior1 = class_prior(data, class1)
    prior2 = class_prior(data, class2)

    return covar, mi_class1, mi_class2, prior1, prior2


def evaluate_model(test_data, covar, mean1, mean2, prior1, prior2):
    """
    Evaluate the given parameters on the test data

    :param test_data: the data to test on
    :param covar: the shared covariance matrix
    :param mean1: the MLE mean for class 1
    :param mean2: the MLE mean for class 2
    :param prior1: the prior probability of class 1
    :param prior2: the prior probability of class 2
    :return: the accuracy rounded to 3 decimals
    """
    # The covariance matrix holds the number of features used
    features = covar.shape[0]
    wrong_predictions = 0
    for i, test_sample in enumerate(test_data['X']):
        actual_class = test_data['C'][i][0]
        predict_class = predict(test_sample[:features], covar, mean1, mean2, prior1, prior2)

        if actual_class == 0 and predict_class > 0.5 or actual_class == 1 and predict_class < 0.5:
            # print("Wrong prediction @ sample [{}] (wanted {}, got {:.3f})".format(i, actual_class, prediction))
            wrong_predictions += 1
    sample_size = test_data['X'].shape[0]
    return (sample_size - wrong_predictions) / sample_size


if __name__ == '__main__':
    # Load the pickle dataset
    vehicle_path = 'HW2/vehicle.pkl'
    vehicle_train, vehicle_test = vehicle.load_data(vehicle_path)

    # Saab=2 and Bus=4 -> try to classify those two
    wanted_classes = [2, 4]
    # Extract features for our wanted classes (train and test set)
    train = vehicle.extract_classes(vehicle_train, wanted_classes)
    test = vehicle.extract_classes(vehicle_test, wanted_classes)

    # Remap the classes to 0/1
    test['C'] = vehicle.rename_class(test['C'], toOne=2, toZero=4)
    train['C'] = vehicle.rename_class(train['C'], toOne=2, toZero=4)

    # Evaluate the accuracy on both the training set and test set
    metrics = [{}, {}]
    print('Evaluating the model for features on the training set ...', end='')
    for feat in range(2, train['X'].shape[1]+1):
        metrics[0]["dataset"] = "train"
        c, m1, m2, p1, p2 = train_model(train, feat)
        acc = evaluate_model(train, c, m1, m2, p1, p2)
        metrics[0][feat] = acc
    print(' done')

    print('Evaluating the model for features on the test set ...', end='')
    for feat in range(2, train['X'].shape[1]+1):
        metrics[1]["dataset"] = "test"
        c, m1, m2, p1, p2 = train_model(test, feat)
        acc = evaluate_model(test, c, m1, m2, p1, p2)
        metrics[1][feat] = acc
    print(' done')

    # Write it to a file
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    if not os.path.exists('images'):
        os.makedirs('images')
    metric_file = 'metrics/genmod.csv'
    image_file = 'images/genmod_acc.png'
    with open(metric_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    # Plot the data into a file
    plotit.plot_csv_metrics(metric_file, image_file)
    print('\nMetrics file: {} \nGraph: {}'.format(metric_file, image_file))

    # Handmade accuracy calculation (test set) for all features
    # feat = 18
    # saab_bus_covar = shared_covariance(train, 2, 4, feat)
    # mi_saab = np.array([mean_mle(train, 2, feat)])
    # mi_bus = np.array([mean_mle(train, 4, feat)])
    # p_saab = class_prior(train, 2)
    # p_bus = class_prior(train, 4)
    #
    # wrong_predictions = 0
    # for i, test_sample in enumerate(test['X']):
    #     actual_class = test['C'][i][0]
    #     prediction = predict(test_sample[:feat], saab_bus_covar, mi_saab, mi_bus, p_saab, p_bus)
    #
    #     if actual_class == 0 and prediction > 0.5 or actual_class == 1 and prediction < 0.5:
    #         # print("Wrong prediction @ sample [{}] (wanted {}, got {:.3f})".format(i, actual_class, prediction))
    #         wrong_predictions += 1
    #
    # samples = test['X'].shape[0]
    # print("Accuracy: {:.3f} ({}/{})".format((samples-wrong_predictions)/samples, samples-wrong_predictions, samples))
    #
