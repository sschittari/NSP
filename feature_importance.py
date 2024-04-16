import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def get_complete_train_test():

    train = pd.read_csv("UNSW_NB15_training-set.csv")

    enc_proto = OneHotEncoder(handle_unknown='ignore')
    trn_proto_enc = enc_proto.fit_transform(np.array(train['proto']).reshape(-1, 1))
    enc_service = OneHotEncoder(handle_unknown='ignore')
    trn_service_enc = enc_service.fit_transform(np.array(train['service']).reshape(-1, 1))
    enc_state = OneHotEncoder(handle_unknown='ignore')
    trn_state_enc = enc_state.fit_transform(np.array(train['state']).reshape(-1, 1))

    train.drop(columns = ['proto', 'service', 'state'], inplace=True)
    train['proto'] = trn_proto_enc.shape[0]
    train['service'] = trn_service_enc.shape[0]
    train['state'] = trn_state_enc.shape[0]

    enc_y = OneHotEncoder(handle_unknown='ignore')
    y_train = enc_y.fit_transform(np.array(train['attack_cat']).reshape(-1, 1)).toarray()

    x_train_df = train
    x_train_df.drop(columns = ['label', 'attack_cat', 'id'], inplace=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.array(x_train_df))
    y_train = np.array(y_train)

    # return x_train, y_train, and x_training pd dataframe
    return x_train, y_train, x_train_df

# TODO: implement optimizing xgboost parameters and average computation for top features
def get_xgboost_top_features():

    print("Getting XGBoost top features...")
    X_trn, y_trn, x_train_df = get_train_test()

    print("Running XGBoost...")
    model = XGBClassifier()
    model.fit(X_trn, y_trn)

    print("Calculating XGBoost feature importance...")
    sorted_idx = model.feature_importances_.argsort()[22:]

    xgboost_top_features = x_train_df.columns[sorted_idx].tolist()
    xgboost_top_features.reverse()
    xgboost_top_features = xgboost_top_features[:20]

    if (False): # True will show feature graph
        plt.barh(x_train_df.columns[sorted_idx], model.feature_importances_[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("XGBoost Feature Ranking")
        plt.show()

    return xgboost_top_features

# TODO: implement random sampling using train_size and test_size
def get_selectfeatures_train_test(xgboost_top_features, train_size, test_size):
    print("Getting train and test split using top features...")

    train = pd.read_csv("UNSW_NB15_training-set.csv")
    test = pd.read_csv("UNSW_NB15_testing-set.csv")

    enc_y = OneHotEncoder(handle_unknown='ignore')
    y_train = enc_y.fit_transform(np.array(train['attack_cat']).reshape(-1, 1)).toarray()

    x_train = train.filter(xgboost_top_features)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.array(x_train))

    y_test = enc_y.transform(np.array(test['attack_cat']).reshape(-1, 1)).toarray()
    x_test = test.filter(xgboost_top_features)
    x_test = scaler.transform(np.array(x_test))

    print("X train shape: " + str(x_train.shape) + ", y train shape:", str(y_train.shape))
    print("X test shape: " + str(x_test.shape) + ", y test shape:", str(y_test.shape))

    return x_train, y_train, x_test, y_test