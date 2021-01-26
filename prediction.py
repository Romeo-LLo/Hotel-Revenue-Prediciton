from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
from Data_preprocess import maxmin_target_encoding_for_adr, maxmin_target_encoding_for_canceled
from sklearn.svm import SVC, SVR

def adr_predict_with_SVR():
    test_size = 14366  # 25% of data; moreover, need to split on the crossing of date
    df_train_o, df_test_o = maxmin_target_encoding_for_adr()

    # split df_train_o to train and validation set
    df_train, df_val = train_test_split(df_train_o, test_size=test_size)

    y_train = df_train.pop('adr')
    x_train = df_train
    y_val = df_val.pop('adr')
    x_val = df_val

    m = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    m.fit(x_train, y_train)
    y_val_pred = m.predict(x_val)

    # After validation, data not split is apply to get the new model
    df_train_y = df_train_o.pop('adr')
    df_train_x = df_train_o

    m.fit(df_train_x, df_train_y)
    y_test_pred = m.predict(df_test_o)

    return y_val_pred, y_test_pred



def canceled_predict_with_SVC():
    df_train_o, df_test_o = maxmin_target_encoding_for_canceled()

    df_train, df_val = train_test_split(df_train_o, test_size=0.25)

    y_train = df_train.pop('is_canceled')
    x_train = df_train
    y_val = df_val.pop('is_canceled')
    x_val = df_val

    df_train_y = df_train_o.pop('is_canceled')
    df_train_x = df_train_o

    # finding the best parameter according to the result of validation
    C = [1, 10, 100, 200]
    K = ['linear', 'poly', 'rbf']
    min_error = 0.5
    best_c = 0.01
    best_k = 'linear'
    for c in C:
        for k in K:
            model = SVC(C=c, kernel=k)
            model.fit(x_train, y_train)
            y_val_pred = model.predict(x_val)
            error = zero_one_loss(y_val, y_val_pred)
            if error < min_error:
                min_error = error
                best_c = c
                best_k = k
    # it is found that best_c = 200 and best_k = rbf
    final_model = SVC(C=best_c, kernel=best_k)
    final_model.fit(df_train_x, df_train_y)
    cancelation = final_model.predict(df_test_o)

    return cancelation

