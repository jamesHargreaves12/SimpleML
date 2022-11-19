import datetime

from tqdm import tqdm
import yaml
from job_orchestration.clientUtils import saveConfigs


def getConfigObject(outputDir:str, partitionNumber:int, config: dict, repeatNumber: int,totalTrainingSize: int, batchSize: int, epochs: int):
    retVal = {
        "configCreated": str(datetime.datetime.now()),
        "outputDir": outputDir,
        "partitionNumber": partitionNumber,
        "repeatNumber": repeatNumber,
        "totalTrainingSize": totalTrainingSize,
        "batchSize": batchSize,
        "epochs": epochs,
        "saveResult": True,
        "tasks": [
            {
                "id": "Train_and_test_no_save",
                "method": "Train_and_test_no_save"
            },
            {
                "id": "writeToS3",
                "method": "WriteToS3"
            }
        ]
    }
    for k in config.keys():
        if k not in retVal:
            retVal[k] = config[k]
    return retVal


def main():
    hyperParamConfig = {('Simple', 100): {'batchSize': 1, 'epochs': 4},
                        ('Simple', 200): {'batchSize': 5, 'epochs': 5},
                        ('Simple', 300): {'batchSize': 10, 'epochs': 5},
                        ('Simple', 400): {'batchSize': 10, 'epochs': 5},
                        ('Simple', 500): {'batchSize': 10, 'epochs': 4},
                        ('Simple', 1000): {'batchSize': 20, 'epochs': 4},
                        ('Simple', 2000): {'batchSize': 5, 'epochs': 4},
                        ('Conv', 100): {'batchSize': 20, 'epochs': 1},
                        ('Conv', 200): {'batchSize': 2, 'epochs': 5},
                        ('Conv', 300): {'batchSize': 3, 'epochs': 5},
                        ('Conv', 400): {'batchSize': 4, 'epochs': 5},
                        ('Conv', 500): {'batchSize': 5, 'epochs': 5},
                        ('Conv', 1000): {'batchSize': 5, 'epochs': 5},
                        ('Conv', 2000): {'batchSize': 5, 'epochs': 5}
                    }
    base_config = yaml.safe_load(open("./baseConfig.yaml"))
    configs = {}
    for repeatNumber in tqdm(range(0,300)):
        for trainSize in (200,300,400,500,2000):
            for modelType in ['Simple','Conv']:
                file_id: str = f"{modelType}_{trainSize}_{repeatNumber}"
                configFileName = 'config_{}.yaml'.format(file_id)
                configs[configFileName] = getConfigObject(
                    outputDir="{date}/" + file_id,
                    partitionNumber=0,
                    config=base_config,
                    repeatNumber=repeatNumber,
                    totalTrainingSize=trainSize,
                    batchSize=hyperParamConfig[(modelType, trainSize)]['batchSize'],
                    epochs = hyperParamConfig[(modelType, trainSize)]['epochs']
                )
    saveConfigs(configs)


def main2():
    # used for budget hyperparameter optimisation stuff
    base_config = yaml.safe_load(open("./baseConfig.yaml"))
    configs = {}
    totalNumberPartitions = 10
    for batchSize in [1,4,16,64,128]:
        base_config["batchSize"] = batchSize
        for epochs in tqdm(range(1,6)):
            base_config["epochs"] = epochs
            id = "hyp_{}_{}_{}_{}".format(batchSize, epochs, base_config['modelType'],
                                         base_config['totalTrainingSize'])
            configFileName = 'config_{}.yaml'.format(id)
            configs[configFileName] = getConfigObject(
                outputDir="{date}/" + id,
                partitionNumber=0,
                config=base_config,
                repeatNumber="{}_{}".format(batchSize,epochs),
            )
    saveConfigs(configs)


if __name__ == "__main__":
    main()
