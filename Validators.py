def train(taskConfig: dict):
    requiredFields = ['partitionNumber', 'totalNumberPartitions', 'modelType', 'repeatNumber', 'outputDir',
                      'totalTrainingSize', 'batchSize', 'epochs']
    errors = []
    for field in requiredFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors


def test(taskConfig: dict):
    requiredFields = ['partitionNumber', 'totalNumberPartitions', 'modelType', 'repeatNumber', 'outputDir']
    errors = []
    for field in requiredFields:
        if field not in taskConfig:
            errors.append("The field {} is required but not provided.".format(field))
    return errors
