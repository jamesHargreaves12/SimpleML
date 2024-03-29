import numpy as np

from keras.datasets import mnist
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from abc import ABCMeta, abstractmethod

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization


class MnistBase(metaclass=ABCMeta):
    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def train(self, xs, ys, N, batchSize, epochs):
        pass

    @abstractmethod
    def getTestOutput(self, xs):
        pass

    @abstractmethod
    def save(self, filepath):
        pass

    @abstractmethod
    def load(self, filepath):
        pass


class SimpleModel(MnistBase):
    def __init__(self):
        # layer 1
        model = Sequential()
        model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # layer 2
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        # output layer
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def summary(self):
        self.model.summary()

    def train(self, xs, ys, N, batchSize, epochs):
        # Prepare data to the expected form
        X_train = xs.reshape(N, 784)
        X_train = X_train.astype('float32')
        X_train /= 255

        Y_train = np_utils.to_categorical(ys, 10)

        self.model.fit(X_train, Y_train,
                       batch_size=batchSize, epochs=epochs,
                       verbose=1)

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        # expect partial as we will only be using it for inference and not training
        self.model.load_weights(filepath).expect_partial()

    def getTestOutput(self, xs):
        X_test = xs.reshape(10000, 784)
        X_test = X_test.astype('float32')
        X_test /= 255

        predicted = self.model.predict(X_test)
        predicted_classes = np.argmax(predicted, axis=1)
        return predicted_classes


class ConvModel(MnistBase):
    def __init__(self):
        model = Sequential()  # Linear stacking of layers

        # Convolution Layer 1
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))  # 32 different 3x3 kernels -- so 32 feature maps
        model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
        convLayer01 = Activation('relu')  # activation
        model.add(convLayer01)

        # Convolution Layer 2
        model.add(Conv2D(32, (3, 3)))  # 32 different 3x3 kernels -- so 32 feature maps
        model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
        model.add(Activation('relu'))  # activation
        convLayer02 = MaxPooling2D(pool_size=(2, 2))  # Pool the max values over a 2x2 kernel
        model.add(convLayer02)

        # Convolution Layer 3
        model.add(Conv2D(64, (3, 3)))  # 64 different 3x3 kernels -- so 64 feature maps
        model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
        convLayer03 = Activation('relu')  # activation
        model.add(convLayer03)

        # Convolution Layer 4
        model.add(Conv2D(64, (3, 3)))  # 64 different 3x3 kernels -- so 64 feature maps
        model.add(BatchNormalization(axis=-1))  # normalize each feature map before activation
        model.add(Activation('relu'))  # activation
        convLayer04 = MaxPooling2D(pool_size=(2, 2))  # Pool the max values over a 2x2 kernel
        model.add(convLayer04)
        model.add(Flatten())  # Flatten final 4x4x64 output matrix into a 1024-length vector

        # Fully Connected Layer 5
        model.add(Dense(512))  # 512 FCN nodes
        model.add(BatchNormalization())  # normalization
        model.add(Activation('relu'))  # activation

        # Fully Connected Layer 6
        model.add(Dropout(0.2))  # 20% dropout of randomly selected nodes
        model.add(Dense(10))  # final 10 FCN nodes
        model.add(Activation('softmax'))  # softmax activation
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def summary(self):
        self.model.summary()

    def train(self, xs, ys, N, batchSize, epochs):
        X_train = xs.reshape(N, 28, 28, 1)  # add an additional dimension to represent the single-channel
        X_train = X_train.astype('float32')  # change integers to 32-bit floating point numbers
        X_train /= 255  # normalize each value for each pixel for the entire vector for each input

        Y_train = np_utils.to_categorical(ys, 10)

        gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                 height_shift_range=0.08, zoom_range=0.08)

        train_generator = gen.flow(X_train, Y_train, batch_size=batchSize)
        self.model.fit(train_generator, steps_per_epoch=N // batchSize, epochs=epochs, verbose=1)

    def getTestOutput(self, xs):
        X_test = xs.reshape(10000, 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255

        predicted = self.model.predict(X_test)
        predicted_classes = np.argmax(predicted, axis=1)
        return predicted_classes

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        # expect partial as we will only be using it for inference and not training
        self.model.load_weights(filepath).expect_partial()


def get_accuracy(preds, real):
    return np.count_nonzero(preds == real) / real.shape[0]


if __name__ == "__main__":
    model = ConvModel()
    model.summary()

    # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()

    print("X_train shape", X_train_real.shape)
    print("y_train shape", y_train_real.shape)
    print("X_test shape", X_test_real.shape)
    print("y_test shape", y_test_real.shape)

    N = X_train_real.shape[0]

    model = ConvModel()
    model.train(X_train_real, y_train_real, N)
    acc = get_accuracy(model.getTestOutput(X_test_real), y_test_real)
    print(acc)
