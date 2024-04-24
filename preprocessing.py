import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import random
import csv

def get_topfeatures_train_test(xgboost_top_features):
    train = pd.read_csv("datasets/UNSW_NB15_training-set.csv")

    enc_y = OneHotEncoder(handle_unknown='ignore')
    y_train = enc_y.fit_transform(np.array(train['attack_cat']).reshape(-1, 1)).toarray()

    X_train = train.filter(xgboost_top_features)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.array(X_train))

    test = pd.read_csv("datasets/UNSW_NB15_testing-set.csv")

    y_test = enc_y.transform(np.array(test['attack_cat']).reshape(-1, 1)).toarray()
    X_test = test.filter(xgboost_top_features)
    X_test = scaler.transform(np.array(X_test))

    return X_train, y_train, X_test, y_test


def generate_train_sampled_csv(X_train, y_train, train_size, seed):

    random.seed(seed)
    train_random_idx = random.sample(range(X_train.shape[0]), train_size)
    X_train_subset = X_train[train_random_idx]
    y_train_subset = y_train[train_random_idx]

    print("Generating samples for train size: " + str(train_size))
    print("X train shape: " + str(X_train_subset.shape) + ", y train shape:", str(y_train_subset.shape))

    # write to csv files
    train_subset = np.concatenate((X_train_subset, y_train_subset), axis=1)
    with open(f'datasets/train_{train_size}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train_subset)


def generate_test_sampled_csv(X_test, y_test, test_size, seed):

    random.seed(seed)
    test_random_idx = random.sample(range(X_test.shape[0]), test_size)
    X_test_subset = X_test[test_random_idx]
    y_test_subset = y_test[test_random_idx]

    print("Generating samples for test size: " + str(test_size))
    print("X test shape: " + str(X_test_subset.shape) + ", y test shape:", str(y_test_subset.shape))

    # write to csv files
    test_subset = np.concatenate((X_test_subset, y_test_subset), axis=1)
    with open(f'datasets/test_{test_size}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test_subset)


def csv_to_xy(file):
    data = np.genfromtxt(file, delimiter=',')

    X = data[:, :20]  # Select first 20 columns for X
    y = data[:, 20:]  # Select remaining columns for Y

    print("Parsed file " + file + " and returned X shape: " + str(X.shape) + ", y shape:", str(y.shape))

    return X, y