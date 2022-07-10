def train(jobConfig, taskConfig):
    requiredJobConfigFields = ['partitionNumber', 'totalNumberPartitions']
    requiredTaskConfigFields = []
    errors = []
    for field in requiredJobConfigFields:
        if field not in jobConfig:
            errors.append("The field {} is required but not provided.".format(field))
    for field in requiredTaskConfigFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors


def createConfigs(jobConfig, taskConfig):
    errors = []
    requiredTaskConfigFields = ['totalNumberPartitions','baseOutputDir','repeatNumber']
    for field in requiredTaskConfigFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors
