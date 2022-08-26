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
                "method": "train"
            },
            {
                "id": "Test",
                "method": "test"
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
    for repeatNumber in tqdm(range(100)):
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

def main2():
    base_config = yaml.safe_load(open("./baseConfig.yaml"))
    configs = {}
    totalNumberPartitions = 10
    for epochs in tqdm(range(1,6)):
        base_config["epochs"] = epochs
        id = "hyp_{}_{}_{}".format(epochs, base_config['modelType'],
                                     base_config['totalTrainingSize'])
        configFileName = 'config_{}.yaml'.format(id)
        configs[configFileName] = getConfigObject(
            outputDir="{date}/" + id,
            partitionNumber=0,
            config=base_config,
            repeatNumber=epochs,
            totalNumberPartitions=totalNumberPartitions
        )
    saveConfigs(configs)


if __name__ == "__main__":
    main2()
