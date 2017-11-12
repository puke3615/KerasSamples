import os
import utils
import numpy as np
import keras.utils
from keras.callbacks import *
from keras.engine import *
from keras.datasets import mnist
from keras.layers import *

n_classes = 10
n_steps = 28
n_feature = 28
n_hidden = 128
batch_size = 200
epochs = 10
validate_rate = 0.2
learning_rate = 1e-2
params_path = utils.root_path('params/rnn.h5')
train = True


def load_data():
    # load data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.divide(x_train, 255.)
    x_test = np.divide(x_test, 255.)
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    # build model
    inputs = Input((n_steps, n_feature))
    x = inputs
    x = LSTM(n_hidden)(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)


def load_weights(model):
    try:
        if os.path.isfile(params_path):
            model.load_weights(params_path)
            print('Load params successfully.')
        else:
            print('Not params found.')
    except:
        os.remove(params_path)
        print('Params error, deleted.')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # load train params if exists
    load_weights(model)

    # train model
    if train:
        utils.ensure_dir(os.path.dirname(params_path))
        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[ModelCheckpoint(params_path)],
            validation_split=validate_rate
        )

    # evaluate model
    scores = model.evaluate(x_test, y_test, batch_size=100)
    print('\n\n')
    print('Loss %s, Accuracy %s' % (scores[0], scores[1]))
    # Loss 0.0673202586091, Accuracy 0.980500007272
