from keras.layers import *
from keras.engine import Model

import nn


def load_data():
    (x_train, y_train), (x_test, y_test) = nn.load_data()
    x_train = np.expand_dims(np.divide(x_train, 255.), -1)
    x_test = np.expand_dims(np.divide(x_test, 255.), -1)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    # build model
    inputs = Input((28, 28, 1))
    x = inputs
    x = Conv2D(16, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)


if __name__ == '__main__':
    nn.MnistNN(
        params_path='params/cnn.h5',
        build_model=build_model,
        load_data=load_data,
    )()
    # Loss 0.0559423817305, Accuracy 0.985800007582
