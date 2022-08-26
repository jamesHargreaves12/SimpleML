import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

from job_orchestration.Config import Config
from job_orchestration.clientUtils import getResults
from job_orchestration.StatusTracker import StatusTracker
from tqdm import tqdm


def StatusFilter(status: StatusTracker):
    return status.start_time > datetime.now() - timedelta(days=5)


def ConfigFilter(config: Config):
    return 'totalNumberPartitions' in config.raw_config \
           and config.raw_config['totalNumberPartitions'] == 10 \
           and config.raw_config['totalTrainingSize'] == 1000 \
           and config.raw_config['modelType'] == 'Simple'


def statisticalSignificance(l: list, r: list, N: int):
    assert (len(l) == len(r))
    total = l + r
    realMedDiff = abs(statistics.median(l) - statistics.median(r))
    biggerCount = 0
    for _ in range(N):
        idx_chosen = set(random.sample(range(2000), k=len(l)))
        l1 = [x for i, x in enumerate(total) if i in idx_chosen]
        r1 = [x for i, x in enumerate(total) if i not in idx_chosen]
        medDiff = abs(statistics.median(l1) - statistics.median(r1))
        if realMedDiff < medDiff:
            biggerCount += 1

    return biggerCount / N


# results = getResults(configFilter=ConfigFilter, statusFilter=StatusFilter)
results = json.load(open("results_simple_N=1000_C=10.json"))  # cached
# print(results)
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

if False:  # box plot
    df2 = pd.DataFrame.from_dict(accs)

    order = sorted([(i, x.median()) for i, x in accs.items()], key=lambda x: x[1])

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    ax = df2[[x for x, _ in order]].plot(kind='box', title='boxplot')

    plt.show()

# Cumulatives
start = min([accs[i].min() for i in range(10)])
end = max([accs[i].max() for i in range(10)])

xs = list(np.arange(start, end, (end - start) / 1000))

for i in [4, 8]:
    ys = [accs[i][accs[i] < x].count() for x in xs]
    plt.plot(xs, ys)
plt.show()

crossCompare = defaultdict(dict)
for i, j in tqdm(product(range(10), range(10))):
    if i == j:
        val = 1
    else:
        val = statisticalSignificance(list(accs[i]), list(accs[j]), 1000)
    crossCompare[i][j] = val

print(crossCompare)
order = [i for i, _ in sorted([(i, x.median()) for i, x in accs.items()], key=lambda x: x[1])]

# head map of statistical significance.
vals = []
for top in order:
    topVals = []
    for bottom in order:
        topVals.append(crossCompare[top][bottom])
    vals.append(topVals)

# 4 worst and 8 best
