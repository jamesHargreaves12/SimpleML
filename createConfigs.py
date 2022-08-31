from tqdm import tqdm
import yaml
from job_orchestration.clientUtils import saveConfigs


def getConfigObject(outputDir, partitionNumber, config: dict, totalNumberPartitions, repeatNumber):
    retVal = {
        "outputDir": outputDir,
        "totalNumberPartitions": totalNumberPartitions,
        "partitionNumber": partitionNumber,
        "repeatNumber": repeatNumber,
        "tasks": [
            {
                "id": "Train",
                "method": "Train"
            },
            {
                "id": "Test",
                "method": "Test"
            },
            {
                "id": "compress",
                "method": "compressModel"
            }
        ]
    }
    for k in config.keys():
        if k not in retVal:
            retVal[k] = config[k]
    return retVal


def main():
    base_config = yaml.safe_load(open("./baseConfig.yaml"))
    configs = {}
    totalNumberPartitions = 10
    for repeatNumber in tqdm(range(295,300)):
        for i in range(totalNumberPartitions):
            id = "{}_{}_{}_{}_{}".format(base_config['modelType'], totalNumberPartitions,
                                         base_config['totalTrainingSize'],
                                         i, repeatNumber)
            configFileName = 'config_{}.yaml'.format(id)
            configs[configFileName] = getConfigObject(
                outputDir= "{date}/"+id,
                partitionNumber=i,
                config=base_config,
                repeatNumber=repeatNumber,
                totalNumberPartitions=totalNumberPartitions
            )
    saveConfigs(configs)

def main2():# used for budget hyperparameter optimisation stuff
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
                totalNumberPartitions=totalNumberPartitions
        )
    saveConfigs(configs)


if __name__ == "__main__":
    main()
