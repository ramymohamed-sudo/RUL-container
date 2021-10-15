
import datetime
import sys
import pandas as pd
from pandas import DataFrame
from pandas import concat
import numpy as np
from random import randint
import requests, zipfile
from io import StringIO
import os
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




start_time = datetime.datetime.now()
file_path = './'
train_file = 'train_FD001.txt'
test_file = 'test_FD001.txt'
rul_file = 'RUL_FD001.txt'


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    # print("n_vars", n_vars)  # 18
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(df.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(df.columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_data(data_path):
    operational_settings = ['os{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['sm{}'.format(i + 1) for i in range(26)]
    cols = ['engine_no', 'cycle'] + operational_settings + sensor_columns
    data = pd.read_csv(data_path, sep=' ', header=None, names=cols)
    data = data.drop(cols[-5:], axis=1)
    # data['index'] = data.index
    # data.index = data['index']
    # data['time'] = pd.date_range('1/1/2000', periods=data.shape[0], freq='600s')

    # print(f'Loaded data with {data.shape[0]} Recordings')    # Recordings\n{} Engines
    # print(f"Number of engines {len(data['engine_no'].unique())}")
    # print('21 Sensor Measurements and 3 Operational Settings')
    return data


if __name__ == "__main__":
    """ 1. Load training data """
    training_data = load_data(file_path+train_file)
    print("data.head():\n", training_data.head())
    print("data.shape:\n", training_data.shape)
    print("data.var():\n", training_data.drop(['engine_no', 'cycle'], axis=1).var())
    num_engine = max(training_data['engine_no'])
    print("num_engine:\n", num_engine)


    """ 2. Load test data """
    test_data = load_data(file_path+test_file)
    test_data.head()
    test_data.shape
    test_data.groupby('engine_no')['cycle'].max()

    # windows size can't be great than max_window_size-2
    max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
    print("max_window_size", max_window_size)
    window_size = max_window_size - 2
    # window_size = 9


    """ 3. Load RUL """
    data_RUL = pd.read_csv(file_path+rul_file,  header=None)
    print("data_RUL.shape", data_RUL.shape)
    print("data_RUL.head()", data_RUL.head())
    num_engine_t = data_RUL.shape[0]
    print("num_engine_t", num_engine_t)



    """ 4. Remove columns that are not useful for prediction """
    training_data.columns
    # Follow variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19 for FD001
    training_data = training_data.drop(['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1) #FD001
    # Follow variable does not have variation, remove os3, sm1, 5, 6, 10, 16, 18, 19
    test_data = test_data.drop(['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1) #FD001
    training_data.head()
    training_data.tail()
    training_data.columns



    """ 5. Define window size. Convert time series to features. """
    # df is the training data
    df_train = pd.DataFrame()

    # For each engine calculate RUL. Loops thru 100 engines in train_FD001.txt
    # Change num_engine if you use other dataset
    # Call series_to_supervised to conver time series to features
    RUL_cap = 130

    for i in range(num_engine):
        df1 = training_data[training_data['engine_no'] == i+1]
        max_cycle = max(df1['cycle'])
        # Calculate Y (RUL)
        # df1['RUL'] = max_cycle - df1['cycle']
        df1['RUL'] = df1['cycle'].apply(lambda x: max_cycle-x)
        # cap RUL to 160 the designed lift
        df1['RUL'] = df1['RUL'].apply(lambda x: RUL_cap if x > RUL_cap else x)
        df2 = df1.drop(['engine_no'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_train = df_train.append(df3)     # dataframes under each other


    print("df_train.head()", df_train.head())
    print("df_train.shape", df_train.shape)
    print("df_train.columns", df_train.columns)

    # df_t is the testing data
    df_test = pd.DataFrame()

    # sys.exit()

    # For each engine calculate RUL. Loops thru 100 engines in test_FD001.txt and RUL_FD001.txt
    # Change num_engine_t if you use other dataset
    for i in range(num_engine_t):
        df1 = test_data[test_data['engine_no'] == i+1]
        max_cycle = max(df1['cycle']) + data_RUL.iloc[i, 0]
        # Calculate Y (RUL)
        df1['RUL'] = max_cycle - df1['cycle']
        df1['RUL'] = df1['RUL'].apply(lambda x: RUL_cap if x > RUL_cap else x)
        # df2 = df1.drop(['engine_no', 'cycle'], axis=1)
        df2 = df1.drop(['engine_no'], axis=1)
        df3 = series_to_supervised(df2, window_size, 1)
        df_test = df_test.append(df3)

    print("df_test.head()", df_test.head())
    print("df_test.shape", df_test.shape)
    print("df_test.columns", df_test.columns)


    df_train.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    df_test.rename(columns={'RUL(t)': 'Y'}, inplace=True)
    # Drop all other RUL columns since they are unknown at time of prediction
    for col in df_train.columns:
        if col.startswith('RUL'):
            df_train.drop([col], axis=1, inplace=True)
            df_test.drop([col], axis=1, inplace=True)


    # df_train.columns
    print("df.shape", df_train.shape)  # 510 feature columns and one RUL columns
    print("df_t.shape", df_test.shape)


    """ 6. Normalize the features X """
    # normalize features to produce a better prediction
    train_values = df_train.drop('Y', axis=1).values  # only normalize X, not y
    # ensure all data is float
    train_values = train_values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_values)
    print("scaled.shape", scaled_train.shape)

    values_test = df_test.drop('Y', axis=1).values  # only normalize X, not y
    values_test = values_test.astype('float32')
    scaled_test = scaler.transform(values_test)
    print("scaled_t.shape", scaled_test.shape)

    # split into train and test sets
    n_train = 100   # 10000
    train_X = scaled_train
    test_X = scaled_test

    # split into input and outputs
    train_y = df_train['Y'].values.astype('float32')
    test_y = df_test['Y'].values.astype('float32')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


    """ 7. Train a LSTM model """
    dim1 = train_X.shape[0]
    dim2 = 1    # for 2-D approach this value is 1
    # for 3-D approach it's the window_size+1
    dim3 = int(train_X.shape[1]/dim2)
    # for 2-D approach this value is train_X.shape[1], for 3-D approach it's train_X.shape[1]/dim2
    print("dim1 and dim2 and dim3", dim1, dim2, dim3)


    # clear tf cache
    tf.keras.backend.clear_session()
    # tf.random.set_seed(51) #tf 2.x
    # tf.random.set_random_seed(71) #tf 1.x
    # np.random.seed(71)
    # TF 1.x
    model = Sequential()
    model.add(LSTM(input_shape=(dim2, dim3), return_sequences=True, units=80))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(40, return_sequences=False))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

    # reshape input to be 3D [samples, timesteps, features]
    history = model.fit(train_X.reshape(dim1, dim2, dim3),
                        train_y, epochs=n_train, batch_size=1000,
                        validation_split=0.1, verbose=2, shuffle=True)
    model.summary()

    # plot history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # 177.02965632527898
    print("min(history.history['val_loss'])", min(history.history['val_loss']))
    history.history['val_loss'].index(min(history.history['val_loss']))

    # epoch > 150
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'][450:], label='train')
    plt.plot(history.history['val_loss'][450:], label='test')
    plt.legend()
    plt.show()

    print("min(history.history['val_loss'])", min(history.history['val_loss']))
    history.history['val_loss'].index(min(history.history['val_loss']))


    # make a prediction
    size_before_reshape = train_X.reshape(dim1, dim2, dim3)
    print("size before and after prediction", train_X.shape, size_before_reshape.shape)
    yhat = model.predict(train_X.reshape(dim1, dim2, dim3))
    model.save('LSTM_model_w_scale.h5')  # creates a HDF5 file 'my_model.h5'
    # !tar -zcvf LSTM_model_w_scale.tgz LSTM_model_w_scale.h5
    # !ls -l

