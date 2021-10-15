
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import keras
import datetime
import sys
from rul_estimator0 import series_to_supervised, load_data
pd.options.mode.chained_assignment = None
from flask import Flask

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask import (url_for, current_app as app)


start_time = datetime.datetime.now()
file_path = './'
# train_file = 'train_FD001.txt'
test_file = 'test_FD001.txt'
rul_file = 'RUL_FD001.txt'
RUL_cap = 130
test_data = load_data(file_path+test_file)
max_window_size = min(test_data.groupby('engine_no')['cycle'].max())
window_size = max_window_size - 2


""" 3. Load RUL """
data_RUL = pd.read_csv(file_path+rul_file,  header=None)
num_engine_t = data_RUL.shape[0]
test_data = test_data.drop(['os3', 'sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19'], axis=1) #FD001

# df_t is the testing data
df_test = pd.DataFrame()
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

df_test.rename(columns={'RUL(t)': 'Y'}, inplace=True)
# Drop all other RUL columns since they are unknown at time of prediction
for col in df_test.columns:
    if col.startswith('RUL'):
        df_test.drop([col], axis=1, inplace=True)

""" Normalization of test data """
scaler = MinMaxScaler(feature_range=(0, 1))
values_test = df_test.drop('Y', axis=1).values  # only normalize X, not y
values_test = values_test.astype('float32')
scaled_test = scaler.fit_transform(values_test)

test_X = scaled_test
test_y = df_test['Y'].values.astype('float32')

dim1 = test_X.shape[0]
dim2 = 1    # for 2-D approach this value is 1
# for 3-D approach it's the window_size+1
dim3 = int(test_X.shape[1]/dim2)

reconstructed_model = keras.models.load_model("LSTM_model_w_scale.h5")
yhat = reconstructed_model.predict(test_X.reshape(dim1, dim2, dim3))

x = test_X.reshape(dim1, dim2, dim3)
# print("shape of test_X.reshape(dim1, dim2, dim3)", x.shape)
# print("shape of yhat", yhat.shape)

app = Flask(__name__)


@app.route("/")
def hello_world():
    return f"The RUL is {str(yhat[0][0])}"


if __name__ == "__main__":
    app.run(host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True)



# app = Flask(__name__)
# @app.route("/")
# def hello_world():
#     return "Hello, World!"

# @app.route("/rul_estimate")
# def hello_world():
#     return "Hello, World!"


# if __name__ == "__main__":
#     app.run(use_reloader=True)



# make a prediction
# testt = train_X.reshape(dim1, dim2, dim3)
# print("size before and after prediction", train_X.shape, testt.shape)
# yhat = model.predict(train_X.reshape(dim1, dim2, dim3))

# reconstructed_model.predict(test_input)




