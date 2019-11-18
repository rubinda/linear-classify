#!/usr/bin/env python3
#
# Probabilistic generative model for predicting vehicle classes
#
# @author David Rubin
import vehicle
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


def mean_mle(data, classNo):
    """
    Calculate the MLE of the mean for a given feature vector of class classNo

    :param data: the vehicle data dict
    :param classNo: the ID of the class we want to calculate the mean for
    :return: vector of means for each feature
    """
    features = vehicle.extract_features(data, classNo)
    feature_sum = np.sum(features, axis=0)
    return feature_sum / len(features)


def shared_covariance(data, class1, class2):
    """
    Compute the MLE of the shared covariance matrix

    :param data: the vehicle data (only for class1 and class2)
    :param class1: the first class ID
    :param class2: the second class ID
    :return: shared covariance matrix (see Appendix on NN19_task2.pdf)
    """
    # Calculate the MLE means
    mean1 = mean_mle(data, class1)
    mean2 = mean_mle(data, class2)
    # Calculate the S matrices
    s1 = s_i(data, class1, mean1)
    s2 = s_i(data, class2, mean2)
    # Get the prior probability for each class
    p1 = class_prior(data, class1)
    p2 = class_prior(data, class2)

    # Return the weighted sum of the matrices for each class
    return np.add(p1 * s1, p2 * s2)


def s_i(data, classNo, mi):
    """
    Compute the S_i from the slides (see Appendix on NN19_task2.pdf)

    :param data: vehicle data
    :param classNo: the class (i) ID
    :param mi: the MLE of the mean
    :return: matrix S_i
    """
    # Extract features for given class only
    features = vehicle.extract_features(data, classNo)

    feature_sum = np.zeros((features[0].shape[0], features[0].shape[0]))
    for feature in features:
        norm_feat = np.array([feature - mi])
        feature_sum = np.add(feature_sum, np.dot(norm_feat.T, norm_feat))
    return np.true_divide(feature_sum, features.shape[0])


if __name__ == '__main__':
    # Load the pickle dataset
    vehicle_path = 'HW2/vehicle.pkl'
    vehicle_train, vehicle_test = vehicle.load_data(vehicle_path)

    # Saab=2 and Bus=4 -> try to classify those two
    wanted_classes = [2, 4]
    # Extract features for our wanted classes (train and test set)
    train = vehicle.extract_classes(vehicle_train, wanted_classes)
    test = vehicle.extract_classes(vehicle_test, wanted_classes)

    #saab_bus_covar = shared_covariance(train, 2, 4)

    # TODO:
    #   implement the posterior distribution guess as on Decision theory slides (page 6)
