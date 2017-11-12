import os
import utils
import numpy as np
import keras.utils
from keras.callbacks import *
from keras.datasets import mnist
from keras.layers import *


def load_data(one_hot=True):
    # load data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if one_hot:
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


class MnistNN:
    def __init__(self, params_path, build_model, load_data=load_data, train=True, batch_size=200, learning_rate=1e-2,
                 validate_rate=0.2,
                 epochs=10):
        self.params_path = utils.root_path(params_path)
        self.load_data = load_data
        self.train = train
        self.batch_size = batch_size
        self.validate_rate = validate_rate
        self.epochs = epochs

        self.model = build_model()
        self.compile_model(learning_rate)

        # load train params if exists
        self.load_weights()

    def compile_model(self, learning_rate):
        # compile model
        optimizer = keras.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def load_weights(self):
        # load weights
        try:
            if os.path.isfile(self.params_path):
                self.model.load_weights(self.params_path)
                print('Load params successfully.')
            else:
                print('Not params found.')
        except:
            os.remove(self.params_path)
            print('Params error, deleted.')

    def __call__(self, *args, **kwargs):
        (x_train, y_train), (x_test, y_test) = self.load_data()
        # train model
        if self.train:
            utils.ensure_dir(os.path.dirname(self.params_path))
            self.model.fit(
                x_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=[ModelCheckpoint(self.params_path)],
                validation_split=self.validate_rate
            )

        # evaluate model
        scores = self.model.evaluate(x_test, y_test, batch_size=100)
        print('\n\n')
        print('Loss %s, Accuracy %s' % (scores[0], scores[1]))
