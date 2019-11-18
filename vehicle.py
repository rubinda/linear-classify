#!/usr/bin/env python3
#
# Loads vehicles from the pickle inside HW2.zip
#
# @author David Rubin

import pickle
import numpy as np
from pathlib import Path


def load_data(vehicle_file):
    """
    Return the data from the pickle file

    :param vehicle_file: string representation of the path to the file
    :return: train, test sets
    """
    # Check if file exists
    filepath = Path(vehicle_file)
    if not filepath.is_file():
        print('Given path <{}> is not a file.'.format(vehicle_file))
        return None, None

    with open(vehicle_file, 'rb') as f:
        vehicle_data = pickle.load(f)
        return vehicle_data['train'], vehicle_data['test']


def extract_classes(data, classNo):
    """
    Extract certain classes (vehicle types) from the test and training data

    :param data
    :param classNo: IDs of the classes we want to extract (array)
    :return: train, test data for the given classes only
    """
    # Mask the indices which hold the given classes and create a new
    # dict which holds only the given classes
    classes = np.array(classNo)
    mask = np.isin(data['C'], classes)
    indices = np.nonzero(mask)[0]
    new_data = {
        'C': data['C'][indices],
        'X': data['X'][indices],
    }
    return new_data


def extract_features(data, classNo):
    """
    Extract the features for the given class

    :param data: vehicle data (dict with classes and features)
    :param classNo: the ID of the chosen class
    :return: array of feature vectors from the specified class
    """
    indices = np.nonzero(data['C'] == classNo)[0]
    return data['X'][indices]
