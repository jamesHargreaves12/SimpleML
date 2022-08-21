import tqdm as tqdm
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
        retVal[k] = config[k]
    return retVal

def main():
    base_config = yaml.safe_load(open("./baseConfig.yaml"))
    configs = {}
    totalNumberPartitions = 10
    for repeatNumber in tqdm(range(500)):
        for i in range(totalNumberPartitions):
            id = "{}_{}_{}_{}_{}".format(base_config['modelType'], totalNumberPartitions,
                                         base_config['totalTrainingSize'],
                                         i, repeatNumber)
            configFileName = 'config_{}.yaml'.format(id)
            configs[configFileName] = getConfigObject(
                outputDir=id,
                partitionNumber=i,
                config=base_config,
                repeatNumber=repeatNumber,
                totalNumberPartitions=totalNumberPartitions
            )
    saveConfigs(configs)

if __name__ == "__main__":
    main()