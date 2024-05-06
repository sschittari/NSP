from feature_importance import get_xgboost_top_features
import preprocessing
import models
import pandas as pd
import numpy as np
import sklearn

GENERATE_TOP_FEATURES = False
GENERATE_SAMPLES = False
TRAIN_MODELS = False

if GENERATE_TOP_FEATURES:
    top_features = get_xgboost_top_features()
else:
    top_features = ['dttl', 'sttl', 'sbytes', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_state_ttl', 
            'ct_srv_dst', 'dbytes', 'tcprtt', 'swin', 'smean', 'dmean', 'sloss', 'ct_src_dport_ltm', 
            'trans_depth', 'response_body_len', 'ct_srv_src', 'dloss', 'spkts', 'is_ftp_login']


if GENERATE_SAMPLES:
    train = pd.read_csv("/dataset/UNSW_NB15_training-set.csv")
    test = pd.read_csv("/dataset/UNSW_NB15_testing-set.csv")

    enc_y = OneHotEncoder(handle_unknown='ignore')
    y_trn = enc_y.fit_transform(np.array(train['attack_cat']).reshape(-1, 1)).toarray()

    x_trn = train.filter(top_features)

    scaler = StandardScaler()
    x_trn = scaler.fit_transform(np.array(x_trn))

    x_trn, x_val, y_trn, y_val = train_test_split(x_trn, y_trn, train_size=0.8)

    y_test = enc_y.fit_transform(np.array(test['attack_cat']).reshape(-1, 1)).toarray()
    x_test = test.filter(top_features)
    x_test = scaler.transform(np.array(x_test))
    
    
else: 
    train_file = 'datasets/train_final.csv'
    test_file = 'datasets/test_final.csv'


if TRAIN_MODELS:
    models.train_DNN(train_file, test_file, 1, 10)
    models.train_CNN(train_file, test_file, 1, 10)
    models.train_CNN_LSTM(train_file, test_file, 1, 10)

print(models.evaluate_saved_model('saved_models/dnn.keras', test_file))
print(models.evaluate_saved_model('saved_models/cnn.keras', test_file))
print(models.evaluate_saved_model('saved_models/cnnlstm.keras', test_file))

