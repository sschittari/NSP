from feature_importance import get_xgboost_top_features
from preprocessing import get_topfeatures_train_test, generate_train_sampled_csv, generate_test_sampled_csv, csv_to_xy
from models import train_DNN, train_CNN, train_CNN_LSTM

# top_features = get_xgboost_top_features() # the xgboost.fit hangs... don't know why
top_features = ['sttl', 'ct_dst_sport_ltm', 'dttl', 'is_sm_ips_ports', 'ct_srv_dst', 'dmean',
                'sbytes', 'ct_state_ttl', 'smean', 'dbytes', 'sloss', 'ct_dst_src_ltm', 'swin',
                'trans_depth', 'ct_flw_http_mthd', 'dloss', 'synack', 'spkts', 'response_body_len',
                'is_ftp_login']

GENERATE_SAMPLES = False

if GENERATE_SAMPLES:
    # generate sampled train files of varying sizes
    X_train, y_train, X_test, y_test = get_topfeatures_train_test(top_features)
    generate_train_sampled_csv(X_train, y_train, 1000, 0)
    generate_train_sampled_csv(X_train, y_train, 5000, 0)
    generate_train_sampled_csv(X_train, y_train, 10000, 0)
    generate_train_sampled_csv(X_train, y_train, 15000, 0)
    generate_train_sampled_csv(X_train, y_train, 20000, 0)

    # generate sampled test file with constant size
    generate_test_sampled_csv(X_test, y_test, 40000, 0)

train_10000 = 'datasets/train_10000.csv'
test_40000 = 'datasets/test_40000.csv'

# train_DNN(train_10000, test_40000)
# train_CNN(train_10000, test_40000)
train_CNN_LSTM(train_10000, test_40000)

