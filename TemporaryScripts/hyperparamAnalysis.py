import yaml
import matplotlib.pyplot as plt


def nth(arr, n):
  return [x[n] for x in arr]

results = yaml.safe_load(open("../results/hyperparam.yaml"))

perfSimple = []
perfConv = []
for modelType in ['Simple','Conv']:
    for tSize in [100,200,300,500, 1000, 2000]:
        print(modelType, tSize)
        ordered = list(sorted([r for r in results if r['modelType']==modelType and r['totalTrainingSize']==tSize], key=lambda x: x['result'], reverse=True ))
        r = ordered[0]
        print(f"    (batchSize={r['batchSize']},epochs={r['epochs']}) = {r['result']}")
        if modelType == 'Simple':
            perfSimple.append((tSize,r['result']))
        else:
            perfConv.append((tSize, r['result']))

plt.plot(nth(perfSimple,0), nth(perfSimple,1))

plt.plot(nth(perfConv,0), nth(perfConv,1))
plt.show()