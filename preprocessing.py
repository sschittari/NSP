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


def csv_to_xy(file):
    data = np.genfromtxt(file, delimiter=',')

    X = data[:, :20]  # Select first 20 columns for X
    y = data[:, 20:]  # Select remaining columns for Y

    print("Parsed file " + file + " and returned X shape: " + str(X.shape) + ", y shape:", str(y.shape))

    return X, y