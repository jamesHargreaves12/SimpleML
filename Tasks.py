import logging
import os
from datetime import datetime

import numpy as np
import yaml
from keras.datasets import mnist
from types import SimpleNamespace

from Models import SimpleModel
from helpers import getPartition


def get_accuracy(preds, real):
    return np.count_nonzero(preds == real) / real.shape[0]


def train(jobConfig, taskConfig):
    logging.info("Start Training")
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()
    (X_train, y_train) = getPartition(jobConfig['partitionNumber'], jobConfig['totalNumberPartitions'], X_train_real,
                                      y_train_real)

    model = SimpleModel()

    model.train(X_train, y_train, X_train.shape[0])
    logging.info("Finished Training")
    path = os.path.join(jobConfig['outputDir'], 'model/postTrain.ckpt')
    model.save(path)
    logging.info("Saved model to " + path)


def test(jobConfig, taskConfig):
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()

    logging.info("Doing test")
    model = SimpleModel()
    path = os.path.join(jobConfig['outputDir'], 'model/postTrain.ckpt')
    logging.info("Loading model from " + path)
    model.load(path)
    acc = get_accuracy(model.getTestOutput(X_test_real), y_test_real)
    logging.info("Resulting accuracy = " + str(acc))
    with open(os.path.join(jobConfig['outputDir'], 'results.yaml'), 'w+') as fp:
        yaml.dump(
            {
                'accuracy': acc,
                'partitionNumber': jobConfig['partitionNumber'],
                'totalNumberPartitions': jobConfig['totalNumberPartitions']
            }, fp)


def getConfigObject(outputDir, partitionNumber, totalNumberPartitions):
    return {
        "outputDir": outputDir,
        "githubRepository": "https://github.com/jamesHargreaves12/SimpleML.git",
        "partitionNumber": partitionNumber,
        "totalNumberPartitions": totalNumberPartitions,
        "tasks": [
            {
                "id": "Train",
                "method": "train"
            },
            {
                "id": "Test",
                "method": "test"
            }
        ]
    }


def createConfigs(jobConfig, taskConfig):
    configs = {}
    for i in range(taskConfig['totalNumberPartitions']):
        configFileName = 'config_{}_{}_{}.yaml'.format(i, taskConfig['totalNumberPartitions'],
                                                       taskConfig['repeatNumber'])
        configs[configFileName] = getConfigObject(
            outputDir=os.path.join(taskConfig['baseOutputDir'],
                                   '{}_{}_{}'.format(i, taskConfig['totalNumberPartitions'],
                                                     taskConfig['repeatNumber'])),
            partitionNumber=i,
            totalNumberPartitions=taskConfig['totalNumberPartitions'])
    return configs


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
