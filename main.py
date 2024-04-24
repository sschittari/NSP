from feature_importance import get_xgboost_top_features
from preprocessing import get_topfeatures_train_test, generate_train_sampled_csv, generate_test_sampled_csv
from DNN import train_DNN
from CNN import train_CNN

# top_features = get_xgboost_top_features() # the xgboost.fit hangs... don't know why
top_features = ['sttl', 'ct_dst_sport_ltm', 'dttl', 'is_sm_ips_ports', 'ct_srv_dst', 'dmean', 'sbytes', 'ct_state_ttl', 'smean', 'dbytes', 'sloss', 'ct_dst_src_ltm', 'swin', 'trans_depth', 'ct_flw_http_mthd', 'dloss', 'synack', 'spkts', 'response_body_len', 'is_ftp_login']

# generate train files of varying sizes
X_train, y_train, X_test, y_test = get_topfeatures_train_test(top_features)
generate_train_sampled_csv(X_train, y_train, 1000, 0)
generate_train_sampled_csv(X_train, y_train, 5000, 0)
generate_train_sampled_csv(X_train, y_train, 10000, 0)
generate_train_sampled_csv(X_train, y_train, 15000, 0)
generate_train_sampled_csv(X_train, y_train, 20000, 0)

# generate test file with constant size
generate_test_sampled_csv(X_test, y_test, 40000, 0)

# dnn_result = train_DNN(X_train, y_train, X_test, y_test)

# cnn_result = train_CNN(X_train, y_train, X_test, y_test)

