import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
from comparator import Comparator as Tester
import matplotlib.pyplot as plt
import numpy

#导入数据
df = pd.read_csv(r"./data/processed_data.csv", engine="python")

tester1 = Tester('SeriousDlqin2yrs')
tester1.addDataset('processed_data', df)

paras = {}
n_estimators = [2, 4, 8, 16, 32, 40, 50, 64, 80, 100, 128, 150, 180]
max_depth = [1, 2, 4, 6, 8, 10, 12, 32, 60, 100, 120, 150]
min_samples_split = np.linspace(2, 1500, 15, endpoint=True)
max_features = ['auto', 'sqrt', 'log2']
min_samples_leaf = np.linspace(1, 500, 15, endpoint=True)
paras["n_estimators"] = n_estimators
paras["max_depth"] = max_depth
paras["max_features"] = max_features
paras["min_samples_split"] = min_samples_split
paras["min_samples_leaf"] = min_samples_leaf

to_tuning = 'min_samples_leaf'

rfc = []
for i in range(0, len(paras[to_tuning])):
    rfc.append(RandomForestClassifier(n_estimators=100, max_depth=16, max_features='auto', min_samples_leaf=int(min_samples_leaf[i])))
    tester1.addModel(i, rfc[i])

test_auc, train_auc, time_spent = tester1.runTests()

# plt.subplot(121)
line1, = plt.plot(paras[to_tuning], train_auc, 'b', label='Train AUC')
line2, = plt.plot(paras[to_tuning], test_auc, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel(to_tuning)
plt.ylim([0.85, 0.9])

# plt.subplot(122)
# line3, = plt.plot(paras[to_tuning], time_spent)
# plt.ylabel('time spent')
# plt.xlabel(to_tuning)
plt.show()



