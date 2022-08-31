import logging
import os
from pathlib import Path

import numpy as np
import yaml
from job_orchestration.TaskBase import TaskBase
from keras.datasets import mnist
from types import SimpleNamespace

from Models import SimpleModel, MnistBase, ConvModel
from helpers import getPartition
from zipModelFile import z7_compress

modelSaveLocation = Path('model/postTrain.ckpt')


class Train(TaskBase):
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        model = SimpleModel() if "modelType" not in self.config or self.config["modelType"] != "Conv" else ConvModel()
        _train(model,
               partitionNumber=self.config['partitionNumber'],
               totalNumberPartitions=self.config['totalNumberPartitions'],
               totalTrainingSize=self.config['totalTrainingSize'],
               batchSize=self.config["batchSize"],
               epochs=self.config["epochs"])
        path = os.path.join(self.config['outputDir'], modelSaveLocation)
        model.save(path)
        logging.info("Saved model to " + path)


def get_accuracy(preds, real):
    return np.count_nonzero(preds == real) / real.shape[0]


def _train(model: MnistBase, partitionNumber, totalNumberPartitions, totalTrainingSize, batchSize, epochs):
    logging.info("Start Training")
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()
    (X_train, y_train) = getPartition(partitionNumber, totalNumberPartitions, X_train_real[:totalTrainingSize],
                                      y_train_real[:totalTrainingSize])

    logging.info("Shape of training Data " + str(X_train.shape))

    model.train(X_train, y_train, X_train.shape[0], batchSize, epochs)
    logging.info("Finished Training")


class Test(TaskBase):
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()

        logging.info("Doing test")
        model = SimpleModel() if "modelType" not in self.config or self.config["modelType"] != "Conv" else ConvModel()
        path = os.path.join(self.config['outputDir'], modelSaveLocation)
        logging.info("Loading model from " + path)
        model.load(path)
        logging.info("Finished loading model")
        acc = get_accuracy(model.getTestOutput(X_test_real), y_test_real)
        logging.info("Resulting accuracy = " + str(acc))
        with open(os.path.join(self.config['outputDir'], 'results.yaml'), 'w+') as fp:
            yaml.dump(
                {
                    'accuracy': acc,
                    'partitionNumber': self.config['partitionNumber'],
                    'modelType': self.config['modelType'],
                    'repeatNumber': self.config['repeatNumber']
                }, fp)
        logging.info("End test")
        return acc  # for hyperparam optimisation we need to return to the lib the value


def compressModel(config: dict):
    modelFolder = os.path.join(config['outputDir'], modelSaveLocation.parent)
    logging.info("Compressing " + str(modelFolder))
    z7_compress(modelFolder)
    logging.info("Finished")


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
    # train(fakeConfig)
    # test(fakeConfig)
