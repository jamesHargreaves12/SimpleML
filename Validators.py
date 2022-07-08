def train(jobConfig, taskConfig):
    requiredJobConfigFields = []
    requiredTaskConfigFields = ['partitionNumber', 'totalNumberPartitions']
    errors = []
    for field in requiredJobConfigFields:
        if field not in jobConfig:
            errors.append("The field {} is required but not provided.".format(field))
    for field in requiredTaskConfigFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors


def createPartitionConfigs(jobConfig, taskConfig):
    errors = []
    requiredTaskConfigFields = ['totalNumberPartitions', 'baseConfigDir','baseOutputDir']
    for field in requiredTaskConfigFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors
