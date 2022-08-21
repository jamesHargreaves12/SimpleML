from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt

from job_orchestration.Config import Config
from job_orchestration.GetResults import getResults
from job_orchestration.StatusTracker import StatusTracker


def StatusFilter(status: StatusTracker):
    return status.start_time > datetime.now() - timedelta(days=1)


def ConfigFilter(config: Config):
    return 'totalNumberPartitions' in config.raw_config \
           and config.raw_config['totalNumberPartitions'] == 10 \
           and config.raw_config['modelType'] == 'Simple'


results = getResults(configFilter=ConfigFilter, statusFilter=StatusFilter)
print("loaded and filtered results")
print(results[0])

cols = defaultdict(list)

res: dict
for res in results:
    for key, val in res.items():
        cols[key].append(val)

df = pd.DataFrame.from_dict(cols)

accs = {}
for i in range(10):
    accs[i] = df[df['partitionNumber'] == i]['accuracy']

df2 = pd.DataFrame.from_dict(accs)

order = sorted([(i, x.median()) for i, x in accs.items()], key=lambda x: x[1])

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
ax = df2[[x for x, _ in order]].plot(kind='box', title='boxplot')

plt.show()
x = 1
