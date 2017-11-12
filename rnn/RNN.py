from keras.layers import *
from keras.engine import Model

import nn


def load_data():
    (x_train, y_train), (x_test, y_test) = nn.load_data()
    x_train = np.divide(x_train, 255.)
    x_test = np.divide(x_test, 255.)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    # build model
    inputs = Input((n_steps, n_feature))
    x = inputs
    x = LSTM(n_hidden)(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)


n_steps = 28
n_feature = 28
n_hidden = 128

if __name__ == '__main__':
    nn.MnistNN(
        params_path='params/rnn.h5',
        build_model=build_model,
        load_data=load_data,
    )()
    # Loss 0.0559423817305, Accuracy 0.985800007582
