import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import numpy
from comparator import Comparator

#导入数据

paras = {}
# 粗调
# n_estimators = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160]
# learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

# 细调
n_estimators = [30, 90, 120, 150, 180, 210, 240, 270, 290, 320]
learning_rate = np.linspace(0.03, 0.1, 8, endpoint=True)


max_depth = [1, 2, 3, 4, 5, 6, 7, 8]
subsample = np.linspace(0.5, 1, 6, endpoint=True)
min_samples_split = np.linspace(750, 1500, 10, endpoint=True)
max_features = ['auto', 'sqrt', 'log2']
loss = ['deviance', 'exponential']
min_samples_leafs = np.linspace(1, 1500, 10, endpoint=True)

paras["n_estimators"] = n_estimators
paras["learning_rate"] = learning_rate
paras["max_depth"] = max_depth
paras["max_features"] = max_features
paras["min_samples_split"] = min_samples_split
paras["min_samples_leafs"] = min_samples_leafs
paras["loss"] = loss
paras["subsample"] = subsample

# 默认参数 ('default GBDT', 'processed_data') --> AUC: 0.8638 (+/- 0.0059)
comparator = Comparator('SeriousDlqin2yrs')
df = pd.read_csv(r"./data/processed_data.csv", engine="python")
comparator.addDataset('processed_data', df)

to_tuning = 'max_depth'
comparator.addModel("1", GradientBoostingClassifier(n_estimators=240, learning_rate=0.06, subsample=0.8, max_depth=5, min_samples_split=1000))
comparator.addModel("2", GradientBoostingClassifier(n_estimators=240, learning_rate=0.1, subsample=0.8, max_depth=5, min_samples_split=1000))
comparator.addModel("3", GradientBoostingClassifier(n_estimators=240, learning_rate=0.1, subsample=0.8, max_depth=5, min_samples_split=1000))
comparator.addModel("4", GradientBoostingClassifier(n_estimators=240, learning_rate=0.05, subsample=0.8, max_depth=6, min_samples_split=1500))
comparator.addModel("5", GradientBoostingClassifier(n_estimators=240, learning_rate=0.1, subsample=0.8, max_depth=6, min_samples_split=1500))
comparator.runTests()

# rfc = []
# for i in range(0, len(paras[to_tuning])):
#     rfc.append(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, subsample=0.8, max_depth=max_depth[i], min_samples_split=750))
#     comparator.addModel(i, rfc[i])
# test_auc, train_auc, time_spent = comparator.runTests()
# line1, = plt.plot(paras[to_tuning], train_auc, 'b', label='Train AUC')
# line2, = plt.plot(paras[to_tuning], test_auc, 'r', label='Test AUC')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('AUC score')
# plt.xlabel(to_tuning)
# plt.ylim([0.845, 0.875])
# plt.show()

for j in range(0, len(learning_rate)):
    rfc = []
    comparator = Comparator('SeriousDlqin2yrs')
    df = pd.read_csv(r"./data/processed_data.csv", engine="python")
    comparator.addDataset('processed_data', df)

    to_tuning = 'n_estimators'

    for i in range(0, len(paras[to_tuning])):
        rfc.append(GradientBoostingClassifier(n_estimators=n_estimators[i], learning_rate=learning_rate[j], subsample=0.8, max_depth=6, min_samples_split=1500))
        comparator.addModel(i, rfc[i])
    test_auc, train_auc, time_spent = comparator.runTests()
    plt.subplot(len(learning_rate)/2, 2, j+1)
    plt.title("learning_rate="+str(learning_rate[j]))
    line1, = plt.plot(paras[to_tuning], train_auc, 'b', label='Train AUC')
    line2, = plt.plot(paras[to_tuning], test_auc, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel(to_tuning)

    plt.ylim([0.85, 0.875])
plt.show()

