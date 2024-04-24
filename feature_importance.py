import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def get_complete_train_test():

    train = pd.read_csv("datasets/UNSW_NB15_training-set.csv")

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

def get_xgboost_top_features():

    TOTAL_RUNS = 5 # number of times to run XGBoost to calculate average for feature importance
    TOP_N_FEATURES = 20 # number of top features to return (and plot)

    aggregate_scores = {}

    print("Running XGBoost...")
    for i in range(TOTAL_RUNS):
        xgb_classifier = XGBClassifier(learning_rate=0.3, max_depth=9, n_estimators=200)
        xgb_classifier.fit(x_train, y_train)

        feature_idx = xgb_classifier.feature_importances_.argsort()
        feature_ratings = xgb_classifier.feature_importances_.tolist()
        feature_ratings.sort(reverse=True)

        for i in range(len(feature_idx)):
            feature_id = feature_idx[i]
            if feature_id not in aggregate_scores:
                aggregate_scores[feature_id] = feature_ratings[i]
            else:
                aggregate_scores[feature_id] += feature_ratings[i]

    print("Calculating feature importance...")
    # sort features by value descending 
    sorted_aggregate_scores = dict(sorted(aggregate_scores.items(), key=lambda x:x[1], reverse=True))

    # compute average score for each feature
    for feature in sorted_aggregate_scores.keys():
        sorted_aggregate_scores[feature] = sorted_aggregate_scores[feature] / TOTAL_RUNS

    print(sorted_aggregate_scores)
    feature_idx_sorted = list(sorted_aggregate_scores.keys())
    feature_ratings_sorted = list(sorted_aggregate_scores.values())

    # get feature names from indices 
    xgboost_top_features = x_train_df.columns[feature_idx_sorted].tolist()
    xgboost_top_features.reverse()
    xgboost_top_features = xgboost_top_features[:TOP_N_FEATURES]
    xgboost_top_feature_scores = feature_ratings_sorted[:TOP_N_FEATURES]

    # generate graph
    plt.barh(xgboost_top_features, xgboost_top_feature_scores)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Ranking")
    plt.savefig('XGBoost_top_features.png')

    return xgboost_top_features