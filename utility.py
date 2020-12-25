import numpy as np
import tensorflow as tf

def prints(content):
    """Prints content with a line gap afterwards

    Args:
        content (any): the content to print
    """
    print(content, '\n')

def supervised_learning_transform(data, start_idx, end_idx, look_back):
    """Transforms time series into supervised learning data.

    Args:
        data (numpy array): the time series to generate data from
        start_idx (int): the first index (inclusive) to generate data from
        end_idx (int): the last index (exclusive) to generate data from
        look_back (int): the look back duration

    Returns:
        x (numpy array): transformed x data
        y (numpy array): transformed y data
    """
    assert start_idx < end_idx
    assert look_back <= start_idx
    assert end_idx <= len(data)

    x, y = [], []
    for i in range(start_idx, end_idx):
        x.append(data[i - look_back:i])
        y.append(data[i])
    x, y = np.array(x), np.array(y)
    x = np.squeeze(x)
    x = x[..., np.newaxis]
    
    assert len(x) == len(y)
    return x, y

def build_rnn_model(n_layers, n_units, input_shape, optimizer):
    """Builds a rnn model.

    Args:
        n_layers (int): number of layers in the rnn
        n_units (list): number of units in each layer of the rnn
        input_shape (tuple): input dimensions to the rnn
        optimizer (string): optimizer for the model

    Returns:
        model (keras model): rnn
    """
    assert len(n_units) == n_layers

    x = inputs = tf.keras.Input(shape=input_shape)
    for i in range(n_layers):
        x = tf.keras.layers.LSTM(n_units[i], return_sequences=(i + 1 < n_layers))(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

def predict_future_values(data, model, scaler, look_back, look_ahead):
    """Uses model and data to predict time series values in the future.

    Args:
        data (numpy array): the time series data
        model (keras model): the model used for prediction
        scaler (scaler obj): the scaler used to scale data
        look_back (int): the look back duration
        look_ahead (int): the look ahead duration 

    Returns:
        y (numpy array): the predicted time series values
    """
    y = []
    x = data[-look_back:].reshape((1, look_back, 1))
    for _ in range(look_ahead):
        pred = model(x).numpy()
        y.append(pred)
        x = np.concatenate([x, np.array([pred]).reshape((1, 1, 1))], axis=1)
        x = x[:, -look_back:]
    y = np.array(y).reshape((1, -1))
    y = scaler.inverse_transform(y)
    return y.reshape(-1)