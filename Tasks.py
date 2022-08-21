import logging
import os

import numpy as np
import yaml
from job_orchestration.Config import TaskConfig
from keras.datasets import mnist
from types import SimpleNamespace

from Models import SimpleModel, MnistBase, ConvModel
from helpers import getPartition

modelSaveLocation = 'model/postTrain.ckpt'


def get_accuracy(preds, real):
    return np.count_nonzero(preds == real) / real.shape[0]


def _train(model: MnistBase, partitionNumber, totalNumberPartitions, totalTrainingSize):
    logging.info("Start Training")
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()
    (X_train, y_train) = getPartition(partitionNumber, totalNumberPartitions, X_train_real[:totalTrainingSize],
                                      y_train_real[:totalTrainingSize])

    logging.info("Shape of training Data " + str(X_train.shape))

    model.train(X_train, y_train, X_train.shape[0])
    logging.info("Finished Training")


def train(taskConfig: TaskConfig):
    model = SimpleModel() if "modelType" not in taskConfig or taskConfig["modelType"] != "Conv" else ConvModel()
    _train(model,
           partitionNumber=taskConfig['partitionNumber'],
           totalNumberPartitions=taskConfig['totalNumberPartitions'],
           totalTrainingSize=taskConfig['totalTrainingSize'])
    path = os.path.join(taskConfig['outputDir'], modelSaveLocation)
    model.save(path)
    logging.info("Saved model to " + path)


def test(config: TaskConfig):
    (X_train_real, y_train_real), (X_test_real, y_test_real) = mnist.load_data()

    logging.info("Doing test")
    model = SimpleModel() if "modelType" not in config or config["modelType"] != "Conv" else ConvModel()
    path = os.path.join(config['outputDir'], modelSaveLocation)
    logging.info("Loading model from " + path)
    model.load(path)
    logging.info("Fished loading model")
    acc = get_accuracy(model.getTestOutput(X_test_real), y_test_real)
    logging.info("Resulting accuracy = " + str(acc))
    with open(os.path.join(config['outputDir'], 'results.yaml'), 'w+') as fp:
        yaml.dump(
            {
                'accuracy': acc,
                'partitionNumber': config['partitionNumber'],
                'modelType': config['modelType'],
                'repeatNumber': config['repeatNumber']
            }, fp)
    logging.info("End test")


def getConfigObject(outputDir, pathToModuleCode, partitionNumber, totalNumberPartitions, modelType, repeatNumber,
                    totalTrainingSize):
    return {
        "outputDir": outputDir,
        "githubRepository": "https://github.com/jamesHargreaves12/SimpleML.git",
        "pathToModuleCode": pathToModuleCode,
        "partitionNumber": partitionNumber,
        "totalNumberPartitions": totalNumberPartitions,
        'repeatNumber': repeatNumber,
        "modelType": modelType,
        "totalTrainingSize": totalTrainingSize,
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


def createConfigs(config: TaskConfig):
    configs = {}
    for i in range(config['totalNumberPartitions']):
        id = "{}_{}_{}_{}_{}".format(config['modelType'], config['totalNumberPartitions'], config['totalTrainingSize'],
                                     i, config['repeatNumber'])
        configFileName = 'config_{}.yaml'.format(id)
        configs[configFileName] = getConfigObject(
            outputDir=os.path.join(config['baseOutputDir'], id),
            partitionNumber=i,
            pathToModuleCode=config['pathToModuleCode'],
            totalNumberPartitions=config['totalNumberPartitions'],
            modelType=config['modelType'],
            repeatNumber=config['repeatNumber'],
            totalTrainingSize=config['totalTrainingSize']
        )
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
    train(fakeConfig)
    test(fakeConfig)
