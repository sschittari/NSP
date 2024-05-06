from feature_importance import get_xgboost_top_features
import preprocessing
import models

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
    # TODO: PREPROCESSING CODE HERE TO GET train_final.csv and test_final.csv
    #
    #
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

