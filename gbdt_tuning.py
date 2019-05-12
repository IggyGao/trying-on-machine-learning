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
n_estimators = [10, 20, 30, 40, 50, 60, 80, 100, 120, 140]
# n_estimators = [10, 30, 50, 60, 80, 100, 120, 140, 180, 200, 220]
learning_rate = [0.07,  0.08,  0.09,  0.10, 0.11]
max_depth = [1, 2, 4, 6, 8, 10, 12, 32, 60, 100]
subsample = []
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
max_features = ['auto', 'sqrt', 'log2']
loss = ['deviance', 'exponential']
min_samples_leafs = np.linspace(1, 1500, 10, endpoint=True)

paras["n_estimators"] = n_estimators
paras["learning_rate"] = learning_rate
paras["max_depth"] = max_depth
paras["max_features"] = max_features
paras["min_samples_splits"] = min_samples_splits
paras["min_samples_leafs"] = min_samples_leafs
paras["loss"] = loss
paras["subsample"] = subsample

# 默认参数 ('default GBDT', 'processed_data') --> AUC: 0.8638 (+/- 0.0059)
comparator = Comparator('SeriousDlqin2yrs')
df = pd.read_csv(r"./data/processed_data.csv", engine="python")
# df = df.drop('Unnamed: 0', axis=1)
comparator.addDataset('processed_data', df)
to_tuning = 'max_depth'
# comparator.addModel("default GBDT", GradientBoostingClassifier(n_estimators=100, learning_rate=0.09))
# comparator.runTests()

rfc = []
for i in range(0, len(paras[to_tuning])):
    rfc.append(GradientBoostingClassifier(n_estimators=100, learning_rate=0.07, loss='exponential', max_depth = max_depth[i]))
    comparator.addModel(i, rfc[i])
test_auc, train_auc, time_spent = comparator.runTests()
line1, = plt.plot(paras[to_tuning], train_auc, 'b', label='Train AUC')
line2, = plt.plot(paras[to_tuning], test_auc, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel(to_tuning)
plt.show()

# for j in range(0, len(learning_rate)):
#     rfc = []
#     comparator = Comparator('SeriousDlqin2yrs')
#     df = pd.read_csv(r"./data/processed_data.csv", engine="python")
#     comparator.addDataset('processed_data', df)
#
#     to_tuning = 'n_estimators'
#
#     for i in range(0, len(paras[to_tuning])):
#         rfc.append(GradientBoostingClassifier(n_estimators=n_estimators[i], learning_rate=learning_rate[j]))
#         comparator.addModel(i, rfc[i])
#     test_auc, train_auc, time_spent = comparator.runTests()
#     plt.subplot(len(learning_rate), 2, 2*j+1)
#     plt.title("learning_rate="+str(learning_rate[j]))
#     line1, = plt.plot(paras[to_tuning], train_auc, 'b', label='Train AUC')
#     line2, = plt.plot(paras[to_tuning], test_auc, 'r', label='Test AUC')
#     plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
#     plt.ylabel('AUC score')
#     plt.xlabel(to_tuning)
#
#     plt.subplot(len(learning_rate), 2, 2*j+2)
#     plt.title("learning_rate="+str(learning_rate[j]))
#     line3 = plt.plot(paras[to_tuning], time_spent, 'y', label='time_spent')
#     plt.ylabel('time_spent')
#     plt.xlabel(to_tuning)
# plt.show()

