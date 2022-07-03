import logging
import os

import numpy as np
import yaml
from keras.datasets import mnist
from types import SimpleNamespace

from Models import SimpleModel


def get_accuracy(preds, real):
    return np.count_nonzero(preds == real) / real.shape[0]


def train(jobConfig, taskConfig):
    logging.info("Start Training")
    model = SimpleModel()

    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()
    model.train(X_train_real, y_train_real)
    logging.info("Finished Training")
    path = os.path.join(jobConfig.outputDir, 'model/postTrain.ckpt')
    model.save(path)
    logging.info("Saved model to " + path)


def test(jobConfig, taskConfig):
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()

    logging.info("Doing test")
    model = SimpleModel()
    path = os.path.join(jobConfig.outputDir, 'model/postTrain.ckpt')
    logging.info("Loading model from " + path)
    model.load(path)
    acc = get_accuracy(model.getTestOutput(X_test_real), y_test_real)
    logging.info("Resulting accuracy = " + str(acc))
    with open(os.path.join(jobConfig.outputDir, 'results.yaml'), 'w+') as fp:
        yaml.dump({'accuracy': acc}, fp)


if __name__ == "__main__":
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    fakeConfigDict = {'outputDir': r'C:\Users\james.hargreaves\PycharmProjects\SimpleMLRepo\Tmp'}
    fakeConfig = SimpleNamespace(**fakeConfigDict)
    train(fakeConfig, {})
    test(fakeConfig, {})
