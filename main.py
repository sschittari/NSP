from feature_importance import get_xgboost_top_features, get_selectfeatures_train_test
from DNN import train_DNN
from CNN import train_CNN

# top_features = get_xgboost_top_features() # the xgboost.fit hangs... don't know why
top_features = ['ct_dst_sport_ltm', 'sttl', 'dttl', 'ct_srv_dst', 'swin', 'sbytes', 'is_sm_ips_ports', 'dmean', 'smean', 'dbytes', 'ct_dst_src_ltm', 'synack', 'trans_depth', 'dpkts', 'sloss', 'dloss', 'ct_state_ttl', 'ct_srv_src', 'response_body_len', 'ct_flw_http_mthd']

X_train, y_train, X_test, y_test = get_selectfeatures_train_test(top_features, 5000, 1000)

print(top_features)

dnn_result = train_DNN(X_train, y_train, X_test, y_test)

cnn_result = train_CNN(X_train, y_train, X_test, y_test)

