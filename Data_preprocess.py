import pandas as pd
import numpy as np
from sklearn import preprocessing
from category_encoders import TargetEncoder


def maxmin_target_encoding_for_adr():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    # drop canceled, no people, negative adr
    df_train = df_train[(df_train.adults + df_train.children + df_train.babies) != 0]  # faster than drop
    df_train = df_train[df_train.adr > 0]
    df_train = df_train[df_train.is_canceled == 0]

    # drop no contribution feature
    df_train.drop(columns=['ID', 'is_canceled', 'reservation_status_date', 'reservation_status'], axis=1, inplace=True)
    df_test.drop(columns=['ID'], axis=1, inplace=True)

    # fill empty cell
    x_train = df_train.copy()
    x_train.fillna(0, inplace=True)
    x_test = df_test.copy()
    x_test.fillna(0, inplace=True)

    # MIN-MAX encoding
    MM_columns = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                  'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                  'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
                  'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests']

    train_features = x_train[MM_columns]
    test_features = x_test[MM_columns]
    combine_features = pd.concat([train_features, test_features])
    Min_Max_Scaler = preprocessing.MinMaxScaler().fit(combine_features)
    train_features = Min_Max_Scaler.transform(train_features)
    test_features = Min_Max_Scaler.transform(test_features)
    x_train[MM_columns] = train_features
    x_test[MM_columns] = test_features

    # Target Encoding
    categorical_cols = ['hotel', 'arrival_date_year', 'arrival_date_month', 'country', 'meal',
                        'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
                        'deposit_type', 'customer_type', 'agent', 'company']
    # let 'adr' range between 0, 1, and named as scaled_adr
    scaled_adr = df_train[['adr']]
    adr_MM_scaler = preprocessing.MinMaxScaler()
    scaled_adr = adr_MM_scaler.fit_transform(scaled_adr)

    # apply scaled_adr to the result of target encoding
    enc = TargetEncoder(cols=categorical_cols)
    enc.fit(x_train[categorical_cols], scaled_adr)
    x_train[categorical_cols] = enc.transform(x_train[categorical_cols])
    x_test[categorical_cols] = enc.transform((x_test[categorical_cols]))

    return x_train, x_test

def maxmin_target_encoding_for_canceled():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    # drop canceled, no people, negative adr
    df_train = df_train[(df_train.adults + df_train.children + df_train.babies) != 0]  # faster than drop
    df_train = df_train[df_train.adr > 0]
    df_train = df_train[(df_train.stays_in_weekend_nights + df_train.stays_in_week_nights) != 0]

    df_train.drop(columns=['ID', 'adr', 'reservation_status_date', 'reservation_status'], axis=1, inplace=True)
    df_test.drop(columns=['ID'], axis=1, inplace=True)

    x_train = df_train.copy()
    x_train.fillna(0, inplace=True)
    x_test = df_test.copy()
    x_test.fillna(0, inplace=True)

    # MIN-MAX encoding
    MM_columns = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
                  'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                  'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
                  'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests']

    train_features = x_train[MM_columns]
    test_features = x_test[MM_columns]
    combine_features = pd.concat([train_features, test_features])
    Min_Max_Scaler = preprocessing.MinMaxScaler().fit(combine_features)
    train_features = Min_Max_Scaler.transform(train_features)
    test_features = Min_Max_Scaler.transform(test_features)
    x_train[MM_columns] = train_features
    x_test[MM_columns] = test_features

    # Target Encoding
    categorical_cols = ['hotel', 'arrival_date_year', 'arrival_date_month', 'meal', 'country',
                        'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type',
                        'deposit_type', 'customer_type', 'agent', 'company']

    enc = TargetEncoder(cols=categorical_cols)
    enc.fit(x_train[categorical_cols], x_train[['is_canceled']])
    x_train[categorical_cols] = enc.transform(x_train[categorical_cols])
    x_test[categorical_cols] = enc.transform((x_test[categorical_cols]))

    return x_train, x_test
