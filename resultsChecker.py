from datetime import datetime, timedelta
import pandas as pd

from job_orchestration.Config import Config
from job_orchestration.GetResults import getResults
from job_orchestration.StatusTracker import StatusTracker


def StatusFilter(status: StatusTracker):
    return status.start_time > datetime.now() - timedelta(days=1)


def ConfigFilter(config: Config):
    return 'totalNumberPartitions' in config.raw_config and config.raw_config['totalNumberPartitions'] == 10


results = getResults(configFilter=ConfigFilter, statusFilter=StatusFilter)

