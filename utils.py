import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

from model import classifier
from constant import *


def time_taken(start, end):
    """Human readable time between `start` and `end`

    :param start: time.time()
    :param end: time.time()

    :returns: day:hour:minute:second.millisecond
    """

    my_time = end-start
    day = my_time // (24 * 3600)
    my_time = my_time % (24 * 3600)
    hour = my_time // 3600
    my_time %= 3600
    minutes = my_time // 60
    my_time %= 60
    seconds = my_time
    milliseconds = ((end - start)-int(end - start))
    day_hour_min_sec = str('%02d' % int(day))+":"+str('%02d' % int(hour))+":"+str('%02d' % int(minutes))+":"+str('%02d' % int(seconds)+"."+str('%.3f' % milliseconds)[2:])

    return day_hour_min_sec


def find_modality_bin_behavior(a_path, db_file_name):
    """
    Finds modality, bins, behavior by using `path` and `dataset` file name

    :param a_path: Dataset path
    :param db_file_name: Dataset file name

    :return: modality, bins, behavior
    """

    modality = a_path.split(os.sep)[1].split("_")[0].capitalize()
    bins = a_path.split(os.sep)[1].split("_")[1]

    if modality == "Proprioception":
        modality = "Haptic"

    if (db_file_name.split(".")[0].split("_")[0]) == 'low':
        behavior = "Drop"
    else:
        behavior = db_file_name.split(".")[0].split("_")[0].capitalize()

    if behavior == "Crush":
        behavior = 'Press'

    return modality, bins, behavior


def reshape_full_data(data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(NUM_OF_CATEGORY, OBJECTS_PER_CATEGORY, TRIALS_PER_OBJECT, -1)


def read_dataset(a_path, db_file_name):
    """
    Read dataset

    :param a_path: Dataset path
    :param db_file_name: Dataset file name

    :return: interaction_data, category_labels, object_labels
    """

    bin_file = open(a_path + os.sep + db_file_name, "rb")
    interaction_data = pickle.load(bin_file)
    category_labels = pickle.load(bin_file)
    object_labels = pickle.load(bin_file)
    bin_file.close()

    return reshape_full_data(interaction_data), reshape_full_data(category_labels), reshape_full_data(object_labels)


def repeat_trials(interaction_data_1_train, interaction_data_2_train):
    """
    Repeat trials for both robots

    :param interaction_data_1_train: Source robot dataset
    :param interaction_data_2_train: Target robot dataset

    :return: Repeated source robot dataset, Repeated target robot dataset
    """

    # Source
    # One example of the source robot can be mapped to all the example of the target robot
    # So, repeating each example of the source robot for each example of target robot
    interaction_data_1_train_repeat = np.repeat(interaction_data_1_train, TRIALS_PER_OBJECT, axis=2)

    # Target
    # Concatenating same examples of target robot to make it same size as source robot
    interaction_data_2_train_repeat = interaction_data_2_train
    for _ in range(TRIALS_PER_OBJECT - 1):
        interaction_data_2_train_repeat = np.concatenate((interaction_data_2_train_repeat, interaction_data_2_train),
                                                         axis=2)

    return interaction_data_1_train_repeat, interaction_data_2_train_repeat


def object_recognition_classifier(clf, data_train, data_test, label_train, label_test, num_of_features):
    """
    Train a classifier and test it based on provided data

    :param clf:
    :param data_train:
    :param data_test:
    :param label_train:
    :param label_test:
    :param num_of_features:

    :return: accuracy, prediction
    """

    train_cats_data = data_train.reshape(-1, num_of_features)
    train_cats_label = label_train.reshape(-1, 1).flatten()

    test_cats_data = data_test.reshape(-1, num_of_features)
    test_cats_label = label_test.reshape(-1, 1).flatten()

    y_acc, y_pred = classifier(clf, train_cats_data, test_cats_data, train_cats_label, test_cats_label)

    return y_acc, y_pred


def print_discretized_data(data, x_values, y_values, modality, behavior, file_path=None):
    """
    prints the data point and save it

    :param data: one data point
    :param x_values: temporal bins
    :param y_values:
    :param modality:
    :param behavior:
    :param file_path:

    :return:
    """
    data = data.reshape(x_values, y_values)

    plt.imshow(data.T)

    title_name = " ".join([behavior, modality, "Features"])
    plt.title(title_name, fontsize=16)
    plt.xlabel("Temporal Bins", fontsize=16)

    if modality == 'Haptic':
        y_label = "Joints"
    elif modality == 'Audio':
        y_label = "Frequency Bins"
    else:
        y_label = ""
    plt.ylabel(y_label, fontsize=16)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, x_values, 1))
    ax.set_yticks(np.arange(0, y_values, 1))
    ax.set_xticklabels(np.arange(1, x_values + 1, 1))
    ax.set_yticklabels(np.arange(1, y_values + 1, 1))

    plt.colorbar()

    if file_path != None:
        plt.savefig(file_path, bbox_inches='tight', dpi=100)

    #plt.show()
    plt.close()


""" Setting 1 """
# Target Robot never interacts with a few categories


def reshape_data_setting1(num_of_category, data):
    """
    Reshape data into (Categories, Objects, Trials)

    :param num_of_category:
    :param data: Dataset list

    :return: reshaped Dataset list
    """
    return data.reshape(num_of_category, OBJECTS_PER_CATEGORY, TRIALS_PER_OBJECT, -1)


def get_data_label_for_given_labels(given_labels, interaction_data, category_labels):
    """
    Get all the examples of the given labels

    :param given_labels: labels to find
    :param interaction_data: examples
    :param category_labels: labels

    :return: Dataset, labels
    """

    data = []
    label = []

    for a_label in given_labels:
        data.append(interaction_data[a_label])
        label.append(category_labels[a_label])

    return np.array(data), np.array(label)


def train_test_splits(num_of_objects):
    """
    Split the data into object based 5 fold cross validation

    :param num_of_objects:

    :return: dictionary containing train test index of 5 folds
    """

    n_folds = 5
    tt_splits = {}

    for a_fold in range(n_folds):
        train_index = []

        test_index = np.arange(a_fold, (a_fold + 1))

        if a_fold > 0:
            train_index.extend(np.arange(0, a_fold))

        if (a_fold + 1) - 1 < num_of_objects - 1:
            train_index.extend(np.arange((a_fold + 1), num_of_objects))

        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("train", []).extend(train_index)
        tt_splits.setdefault("fold_" + str(a_fold), {}).setdefault("test", []).extend(test_index)

    return tt_splits


def object_based_5_fold_cross_validation(clf, data_train, data_test, labels, num_of_features):
    """
    Perform object based 5 fold cross validation and return mean accuracy

    :param clf: classifier
    :param data_train: Training dataset
    :param data_test: Testing dataset
    :param labels: True labels
    :param num_of_features: Number of features of the robot

    :return: mean accuracy of 5 fold validation
    """

    tts = train_test_splits(OBJECTS_PER_CATEGORY)

    my_acc = []

    for a_fold in sorted(tts):
        train_cats_index = tts[a_fold]["train"]
        test_cats_index = tts[a_fold]["test"]

        train_cats_data = data_train[:, train_cats_index]
        train_cats_label = labels[:, train_cats_index]
        train_cats_data = train_cats_data.reshape(-1, num_of_features)
        train_cats_label = train_cats_label.reshape(-1, 1).flatten()

        test_cats_data = data_test[:, test_cats_index]
        test_cats_label = labels[:, test_cats_index]
        test_cats_data = test_cats_data.reshape(-1, num_of_features)
        test_cats_label = test_cats_label.reshape(-1, 1).flatten()

        y_acc, y_pred = classifier(clf, train_cats_data, test_cats_data, train_cats_label, test_cats_label)
        my_acc.append(y_acc)

    return np.mean(my_acc)


