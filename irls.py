#!/usr/bin/env python3
#
# The IRLS (Iteratively reweighted least squares) algorithm for linear classification
#
# @author David Rubin
import vehicle
import genmod
import plotit
import os
import json
import numpy as np


def init_weights(n, low=-0.5, hi=0.5):
    """
    Initialize n random weights in an interval [low, hi)
    :param n: the number of weights to be generated
    :param low: the low boundary
    :param hi: the high boundary
    :return: array of n random numbers from interval [low, hi)
    """
    return np.random.uniform(low, hi, n).reshape(n, 1)


def expected_bernoulli(w, x):
    """
    The expected value of a Bernoulli distribution

    :param w: weights vector
    :param x: features vector
    :return: the expected value {0, 1}
    """
    return 1 / (1 + np.exp(np.dot(-w.T, x)))


def expected_classes(X, w):
    """
    Returns an array with expected classes for each row in X given weights w
    :param X: feature matrix for all samples
    :param w: vector of weights for a single sample
    :return: array with expected values (classifications) for each sample
    """
    return np.array([expected_bernoulli(w, X[i]) for i in range(X.shape[0])]).reshape(X.shape[0], 1)


def make_predictions(X, w):
    """
    Make predictions for each sample (row in X) using the sigmoid function (on w.T * x)
    :param X: feature matrix
    :param w: weights vector
    :return: array of predicted classes (between 0 and 1)
    """
    return np.array([genmod.sigmoid(np.dot(w.T, sample)) for sample in X]).reshape(X.shape[0], 1)


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def cee(predictions, targets):
    """ Cross entropy error function """
    return np.sum([-np.log(y) if t else -np.log(1-y) for t, y in zip(targets, predictions)])


def mcr(predictions, targets):
    """ The misclassification rate """
    return np.mean(np.abs(np.around(predictions) - targets))


def irls(X, w0, t, metric, X_test, t_test, maxiter=100):
    """
    The IRLS algorithm.

    Update rule w_{new}=w_{old}-(X^T R X)^{-1} X^T (y-t), where:
    X   ... feature matrix
    R   ... diagonal matrix of expected values (S on Wikipedia)
    y   ... vector of response values (output of the sigmoid function for w.T dot x)
    t   ... targets for each row

    :param X: the feature matrix (should have 1 prepended in each row for the bias)
    :param w0: the initial weights (see init_weights)
    :param t: targets (from the training set)
    :param metric: The metric dict in which we store the misclassification rates and cross
     entropy errors for each epoch
    :param X_test: the test X (used only for metrics over epochs)
    :param t_test: the test targets (used only for metrics over epochs)
    :param maxiter: maximum number of iterations, in case the stopping condition doesn't fire
    :return: adjusted weights for better classification
    """
    w_old = np.copy(w0)
    for i in range(maxiter):
        y = make_predictions(X, w_old)
        # Calculate the metrics and store them
        train_cee = cee(y, t)
        train_mis = mcr(y, t)
        metric['train-misclass_rates'].append(train_mis)
        metric['train-cees'].append(train_cee)
        _y = make_predictions(X_test, w_old)
        test_cee = cee(_y, t_test)
        test_mis = mcr(_y, t_test)
        metric['test-misclass_rates'].append(test_mis)
        metric['test-cees'].append(test_cee)

        # Our stopping condition is the CEE becoming low (overfitting  possible?)
        if train_cee < 10**-6:
            # print("Stopping because of low CEE @ {} iterations".format(i))
            break

        R = np.diag(np.multiply(y, 1 - y).T[0])
        H = np.dot(np.dot(X.T, R), X)

        # Our stopping condition is also H becoming singular
        if not np.isfinite(np.linalg.cond(H)):
            # print("Matrix was becoming singular".format(i))
            break
        H = np.linalg.inv(H)
        D = np.dot(X.T, (y - t))
        delta = np.matmul(H, D)
        w_new = w_old - delta

        # Calculate the norm and compare it with tolerance (?)
        #  Stop when (norm(w_new - w_old) / (1 + norm(w_old)) < tol)
        # if np.linalg.norm(w_new - w_old, 1) / (1 + np.linalg.norm(w_old)) < 0.001:
        #     print("Tolerance is smaller breaking {}".format(i))
        #     break
        w_old = w_new
    return w_old


def get_accuracy(weights, X, targets):
    """
    Calculates the accuracy of the IRLS weights

    :param weights: IRLS weights (bias included)
    :param X: the feature set
    :param targets: the target classes
    :return: accuracy [0-1]
    """
    predictions = make_predictions(X, weights)
    wrong_predictions = 0
    for p, y in zip(predictions, targets):
        p = np.around(p)
        if p != y[0]:
            wrong_predictions += 1
    samples = X.shape[0]
    return (samples - wrong_predictions) / samples


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
    train['C'] = vehicle.rename_class(train['C'], toOne=2, toZero=4)
    test['C'] = vehicle.rename_class(test['C'], toOne=2, toZero=4)
    # Normalize the feature data
    # train['X'] = vehicle.normalize_features(train['X'])

    # We add a 1 to every feature set to compensate for the bias
    train['X'] = np.insert(train['X'], 0, 1, axis=1)
    test['X'] = np.insert(test['X'], 0, 1, axis=1)
    # The maximum number of features we can use (including the bias)
    max_features = train['X'].shape[1]

    metrics = []
    print("Recording metrics over the training and test set for [2..{}] features ...".format(max_features-1), end='')
    # We start at 3 (2 features + bias) and end at max_features (bias included)
    for feat in range(3, max_features+1):
        # Get some initial weights
        w_init = init_weights(feat, low=-0.000001, hi=0.000001)

        # Prepare a metric object we will dump into a JSON file
        current_metric = {
            'train-cees': [],
            'train-misclass_rates': [],
            'test-cees': [],
            'test-misclass_rates': [],
            'features_used': feat,
            'accuracy': 0,
            'initial_weights': w_init.tolist(),
        }
        # Train The model for only a certain amount of features
        # During the training phase record the Cross entropy error and misclassification
        # rate into current_metric
        weights = irls(train['X'][:, :feat], w_init, train['C'], current_metric, test['X'][:, :feat], test['C'])
        # Test the accuracy of the final model
        current_metric['accuracy'] = get_accuracy(weights, test['X'][:, :feat], test['C'])
        metrics.append(current_metric)
    print(' done')

    # Create the folders for the metrics and images (if not exists)
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    if not os.path.exists('images'):
        os.makedirs('images')

    json_file = 'metrics/irls.json'
    with open(json_file, 'w') as f:
        json.dump(metrics, f)
        print('Metrics stored into {}'.format(json_file))

    # Create the graphs
    plotit.plot_json_metrics(json_file, 'images/irls_acc.png', 'images/irls_cee.png', 'images/irls_mcr.png')
    print('Graphs stored into /images/irls_*.png')

    # Get the handmade accuracy on the test set (using all features)
    # ps = make_predictions(test['X'], weights)
    # wrong_predictions = 0
    # for p, actual in zip(ps, test['C']):
    #     # Our guesses are from a sigmoid function, round them to the closest value (0 or 1)
    #     p = np.around(p)
    #     if p != actual[0]:
    #         wrong_predictions += 1
    #
    # samples = test['X'].shape[0]
    # print("Accuracy: {:.3f} ({}/{})".format(
    #     (samples - wrong_predictions) / samples,
    #     samples - wrong_predictions,
    #     samples))

