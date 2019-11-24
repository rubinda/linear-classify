#!/usr/bin/env python3
#
# Contains plot helper function for the .CSV files containing metrics of IRLS and PGM
#
# @author David Rubin
import csv
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


def read_csv_data(filename):
    """
    Read the metrics from a file into a dict
    :param filename: the file containing the metrics
    :return: list of dicts
    """
    filepath = Path(filename)
    read_lines = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        read_lines = [row for row in reader]
    return read_lines


def read_json_data(filename):
    """
    Read the metrics stored in a JSON file into a list of dicts
    :param filename: the JSON metrics (from irls.py)
    :return: list of dicts
    """
    filepath = Path(filename)
    read_lines = []
    with open(filepath, 'r') as f:
        read_lines = json.load(f)
    return read_lines


def plot_csv_metrics(metrics_file, save_file):
    """
    Plot the given matrics and save the graph to the given file

    :param metrics_file: the metrics from read_data
    :param save_file: filename for the graph image to be stored
    """
    metrics = read_csv_data(metrics_file)
    plt.xlabel('Features used')
    plt.ylabel('Accuracy')
    for metric in metrics:
        line_label = metric['dataset']
        x = np.array(list(metric.keys()))[1:]
        y = np.array(list(metric.values()))[1:]
        plt.plot(x.astype(int), y.astype(float), label=line_label)
    plt.legend()
    plt.savefig(save_file)


def plot_json_metrics(metrics_file, acc_graph, cee_graph, mcr_graph):
    """
    Plot the given metrics and save some graphs to files

    :param metrics_file: the json file that holds metrics
    :param acc_graph: filename for storing the accuracy graph
    :param cee_graph: filename for storing the cross entropy error graph
    :param mcr_graph: filename for storing the misclassification rate graph
    """
    metrics = read_json_data(metrics_file)

    # Plot the accuracies
    accuracies = np.array([metric['accuracy'] for metric in metrics], dtype=float)
    features = np.array([metric['features_used'] for metric in metrics], dtype=int)
    plt.xlabel('Features used')
    plt.ylabel('Accuracy')
    plt.plot(features, accuracies)
    plt.savefig(acc_graph)
    plt.clf()

    # Plot the cross entropy error over epochs for some of the features used
    # I have chosen to take 3 feature samples: 2, 9 & 18
    plt.xlabel('Training epoch')
    plt.ylabel('Cross entropy error')
    cee_path = Path(cee_graph)

    # 2 features used (the metrics starts at 2 samples + 1 bias = 3 features_used)
    ep2 = np.arange(1, len(metrics[0]['train-cees'])+1)
    cee2_test = np.array(metrics[0]['test-cees'], dtype=float)
    cee2_train = np.array(metrics[0]['train-cees'], dtype=float)
    plt.plot(ep2, cee2_test, label='CEE Test (feat=2)')
    plt.plot(ep2, cee2_train, label='CEE Train (feat=2)')
    plt.legend()
    plt.savefig(Path(str(cee_path.with_suffix('')) + "-2.png"))
    plt.clf()
    # 9 features used
    plt.xlabel('Training epoch')
    plt.ylabel('Cross entropy error')
    ep9 = np.arange(1, len(metrics[7]['train-cees'])+1)
    cee9_test = np.array(metrics[7]['test-cees'], dtype=float)
    cee9_train = np.array(metrics[7]['train-cees'], dtype=float)
    plt.plot(ep9, cee9_test, label='CEE Test (feat=9)')
    plt.plot(ep9, cee9_train, label='CEE Train (feat=9)')
    plt.legend()
    plt.savefig(Path(str(cee_path.with_suffix('')) + "-9.png"))
    plt.clf()
    # 18 features used
    plt.xlabel('Training epoch')
    plt.ylabel('Cross entropy error')
    ep18 = np.arange(1, len(metrics[16]['train-cees']) + 1)
    cee18_test = np.array(metrics[16]['test-cees'], dtype=float)
    cee18_train = np.array(metrics[16]['train-cees'], dtype=float)
    plt.plot(ep18, cee18_test, label='CEE Test (feat=18)')
    plt.plot(ep18, cee18_train, label='CEE Train (feat=18)')
    plt.legend()
    plt.savefig(Path(str(cee_path.with_suffix('')) + "-18.png"))
    plt.clf()

    # Plot the misclassification rates for a selection of features used
    plt.xlabel('Training epoch')
    plt.ylabel('Misclassification rate')
    mcr_path = Path(mcr_graph)

    # 2 features used (the metrics starts at 2 samples + 1 bias = 3 features_used)
    ep2 = np.arange(1, len(metrics[0]['train-misclass_rates']) + 1)
    mcr2_test = np.array(metrics[0]['test-misclass_rates'], dtype=float)
    mcr2_train = np.array(metrics[0]['train-misclass_rates'], dtype=float)
    plt.plot(ep2, mcr2_test, label='CEE Test (feat=2)')
    plt.plot(ep2, mcr2_train, label='CEE Train (feat=2)')
    plt.legend()
    plt.savefig(Path(str(mcr_path.with_suffix('')) + "-2.png"))
    plt.clf()
    # 9 features used
    plt.xlabel('Training epoch')
    plt.ylabel('Misclassification rate')
    ep9 = np.arange(1, len(metrics[7]['train-misclass_rates']) + 1)
    mcr9_test = np.array(metrics[7]['test-misclass_rates'], dtype=float)
    mcr9_train = np.array(metrics[7]['train-misclass_rates'], dtype=float)
    plt.plot(ep9, mcr9_test, label='CEE Test (feat=9)')
    plt.plot(ep9, mcr9_train, label='CEE Train (feat=9)')
    plt.legend()
    plt.savefig(Path(str(mcr_path.with_suffix('')) + "-9.png"))
    plt.clf()
    # 18 features used
    plt.xlabel('Training epoch')
    plt.ylabel('Misclassification rate')
    ep18 = np.arange(1, len(metrics[16]['train-misclass_rates']) + 1)
    mcr18_test = np.array(metrics[16]['test-misclass_rates'], dtype=float)
    mcr18_train = np.array(metrics[16]['train-misclass_rates'], dtype=float)
    plt.plot(ep18, mcr18_test, label='CEE Test (feat=18)')
    plt.plot(ep18, mcr18_train, label='CEE Train (feat=18)')
    plt.legend()
    plt.savefig(Path(str(mcr_path.with_suffix('')) + "-18.png"))
    pass


if __name__ == '__main__':
    # plot_csv_metrics('metrics/genmod.csv', 'images/genmod_acc.png')
    plot_json_metrics('metrics/irls.json', 'images/irls_acc.png', 'images/irls_cee.png', 'images/irls_mcr.png')
    print('done')