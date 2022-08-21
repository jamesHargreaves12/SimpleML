from job_orchestration.Config import TaskConfig


def train(taskConfig:TaskConfig):
    requiredFields = ['partitionNumber', 'totalNumberPartitions','modelType','repeatNumber']
    errors = []
    for field in requiredFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors


def createConfigs(taskConfig:TaskConfig):
    errors = []
    requiredTaskConfigFields = ['totalNumberPartitions','baseOutputDir','repeatNumber','modelType']
    for field in requiredTaskConfigFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors
