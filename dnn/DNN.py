from keras.layers import *
from keras.engine import Model

import nn


def load_data():
    (x_train, y_train), (x_test, y_test) = nn.load_data()
    x_train = np.divide(x_train, 255.).reshape([-1, 784])
    x_test = np.divide(x_test, 255.).reshape([-1, 784])
    return (x_train, y_train), (x_test, y_test)


def build_model():
    # build model
    inputs = Input((784,))
    x = inputs
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)


if __name__ == '__main__':
    nn.MnistNN(
        params_path='params/dnn.h5',
        build_model=build_model,
        load_data=load_data,
    )()
    # Loss 0.171557076013, Accuracy 0.964600001574
