import logging
import os
from pathlib import Path

import numpy as np
import yaml
from job_orchestration.TaskBase import TaskBase
from job_orchestration.configBase import TaskWithInitAndValidate
from keras.datasets import mnist
from types import SimpleNamespace

from Models import SimpleModel, ConvModel
from helpers import getPartition
from zipModelFile import z7_compress

modelSaveLocation = Path('model/postTrain.ckpt')


class Train(TaskWithInitAndValidate):
    partitionNumber: int
    totalNumberPartitions: int
    modelType: str # nullable
    repeatNumber: int
    outputDir: str
    totalTrainingSize: int
    batchSize: int
    epochs: int

    def run(self):
        model = SimpleModel() if self.modelType != "Conv" else ConvModel()

        logging.info("Start Training")
        (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()
        (X_train, y_train) = getPartition(self.partitionNumber, self.totalNumberPartitions, X_train_real[:self.totalTrainingSize],
                                          y_train_real[:self.totalTrainingSize])

        logging.info("Shape of training Data " + str(X_train.shape))

        model.train(X_train, y_train, X_train.shape[0], self.batchSize, self.epochs)
        logging.info("Finished Training")

        path = os.path.join(self.outputDir, modelSaveLocation)
        model.save(path)
        logging.info("Saved model to " + path)


class Test(TaskWithInitAndValidate):
    partitionNumber: int
    modelType: str
    repeatNumber: int
    outputDir: str

    def run(self):
        (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()

        logging.info("Doing test")
        model = SimpleModel() if self.modelType != "Conv" else ConvModel()
        path = os.path.join(self.outputDir, modelSaveLocation)
        logging.info("Loading model from " + path)
        model.load(path)
        logging.info("Finished loading model")
        acc = get_accuracy(model.getTestOutput(X_test_real), y_test_real)
        logging.info("Resulting accuracy = " + str(acc))
        with open(os.path.join(self.outputDir, 'results.yaml'), 'w+') as fp:
            yaml.dump(
                {
                    'accuracy': acc,
                    'partitionNumber': self.partitionNumber,
                    'modelType': self.modelType,
                    'repeatNumber': self.repeatNumber
                }, fp)
        logging.info("End test")
        return acc  # for hyperparam optimisation we need to return to the lib the value


def get_accuracy(preds, real):
    return np.count_nonzero(preds == real) / real.shape[0]


class CompressModel(TaskWithInitAndValidate):
    outputDir: str

    def run(self):
        modelFolder = os.path.join(self.outputDir, modelSaveLocation.parent)
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
