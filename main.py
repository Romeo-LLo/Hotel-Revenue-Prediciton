import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from ML_model import canceled_predict_with_SVC, adr_predict_with_SVR

def Test_toLabel_with_canceled():

    df_test = pd.read_csv('test.csv')
    df_test_nolabel = pd.read_csv('test_nolabel.csv')

    _, y_test_pred = adr_predict_with_SVR()
    cancelation = canceled_predict_with_SVC()

    # the predicted adr on each row is multiplied by "total stays day", is canceled or not is also take into consideration
    # cancelation: 0 -> not canceled ----> after modified----> -(0)+1 = 1
    #              1 -> canceled     ----> after modified----> -(1)+1 = 0
    df_test['daily_ravenue'] = (df_test['stays_in_weekend_nights'] + df_test['stays_in_week_nights']) * y_test_pred * (np.negative(cancelation) + 1)

    # obtain the daily revenue
    sum = df_test.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])['daily_ravenue'].sum().reset_index()

    #　map daily revenue to the label
    df_test_nolabel['label'] = sum['daily_ravenue'] // 10000
    df_test_nolabel.to_csv("note/test_label", index=False)


def Val_toLabel():
    y_val_pred, _= adr_predict_with_SVR()

    test_size = 14366
    val_remove_num = 42981
    val_romove_day = 482

    df_train = pd.read_csv('train.csv')
    df_train_label = pd.read_csv('train_label.csv')

    df_train = df_train[(df_train.adults + df_train.children + df_train.babies) != 0]
    df_train = df_train[df_train.is_canceled == 0]
    df_train = df_train[df_train.adr > 0]

    df_train.drop(df_train.index[:val_remove_num], inplace=True)
    df_train_label.drop(df_train_label.index[:val_romove_day], inplace=True)

    df_train['daily_ravenue'] = (df_train['stays_in_weekend_nights'] + df_train['stays_in_week_nights']) * y_val_pred

    sum = df_train.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])['daily_ravenue'].sum().reset_index()

    df_train_label['label'] = sum['daily_ravenue'] // 10000
    df_train_label.to_csv("note/val_label.csv", index=False)



if __name__ == '__main__':
    # Test_toLabel_with_canceled()
    Val_toLabel()
