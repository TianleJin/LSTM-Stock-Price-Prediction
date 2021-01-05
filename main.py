import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from utility import prints, supervised_learning_transform, build_rnn_model, predict_future_values
tf.keras.backend.set_floatx('float64')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# reading input file
df = pd.read_csv('./Data/Z1P.AX.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df = df.set_index('Date')
prints('Sampling data...')
prints(df.head())
prints('Retrieving data info...')
df.info()
print()


# benchmark mse
truth = df['Close']
prediction = df['Close'].shift()
prediction = prediction.fillna(method='backfill')
mse = mean_squared_error(truth, prediction)
prints('Calculating benchmark mse...')
prints(mse)


# extract raw data
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)


# data parameters
look_back = 7
look_ahead = 10
train_size = int(0.60 * len(data))
valid_size = int(0.20 * len(data))


# synthesis training data
x_train, y_train = supervised_learning_transform(data, look_back, train_size, look_back)
x_valid, y_valid = supervised_learning_transform(data, train_size, train_size + valid_size, look_back)
x_test, y_test = supervised_learning_transform(data, train_size + valid_size, len(data), look_back)
prints('Retrieving shape of data...')
prints('Size of training data: {}'.format(x_train.shape))
prints('Size of validation data: {}'.format(x_valid.shape))
prints('Size of testing data: {}'.format(x_test.shape))


# create model
model = build_rnn_model(
    n_layers=2,
    n_units=[50, 50],
    input_shape=(x_train.shape[1], 1),
    optimizer='adam'
)
prints('Summarising model...')
model.summary()
print()


# fit model
history = model.fit(
    x_train, 
    y_train, 
    epochs=15, 
    batch_size=len(x_train), 
    validation_data=(x_valid, y_valid),
    shuffle=False
)


# display training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend()
plt.show()


# name model
train_loss = float(model.evaluate(x_train, y_train))
valid_loss = float(model.evaluate(x_valid, y_valid))
test_loss = float(model.evaluate(x_test, y_test))
model_name = 'model_train_loss_{:.6f}_val_loss_{:.6f}_test_loss_{:.6f}'.format(
    train_loss, 
    valid_loss, 
    test_loss
)
prints('Naming model...')
prints(model_name)


# save model
save_dir = os.path.join(os.getcwd(), 'models', model_name)
save_path = os.path.join(save_dir, 'ckpt')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save_weights(save_path)
prints('Saving model...')
prints(save_path)


# display raw data vs predicted data
train_dates = df.index[look_back: train_size]
valid_dates = df.index[train_size: train_size + valid_size]
test_dates = df.index[train_size + valid_size:]
future_dates = pd.date_range(start=df.index[-1], periods=look_ahead+1)[1:]

train_preds = scaler.inverse_transform(model(x_train))
valid_preds = scaler.inverse_transform(model(x_valid))
test_preds = scaler.inverse_transform(model(x_test))
future_preds = predict_future_values(data, model, scaler, look_back, look_ahead)

fig = plt.figure()
plt.plot(df.index, df['Close'], label='ground truth')
plt.plot(train_dates, train_preds, label='train prediction')
plt.plot(valid_dates, valid_preds, label='valid prediction')
plt.plot(test_dates, test_preds, label='test prediction')
plt.plot(future_dates, future_preds, label='future prediction')
plt.legend()
plt.show()


# future summary
prints('Displaying predictions...')
prints(pd.DataFrame(future_preds, columns=['Close'], index=future_dates))