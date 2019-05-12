import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
from comparator import Comparator as Tester
import matplotlib.pyplot as plt
import numpy

#导入数据
df = pd.read_csv(r"./data/cs-training.csv", engine="python")

mode = df["MonthlyIncome"].median()
# mode = df["MonthlyIncome"].mode().iloc[0]
df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = mode

# 删除比较少的缺失值
df = df.dropna()

# 删除重复项
df = df.drop_duplicates()

# 1.删除DebtRatio 异常值
removed_debt_outliers = df.drop(df[df['DebtRatio'] > 3489.025].index)

# 2.删除RevolvingUtilizationOfUnsecuredLines 异常值
dfus = df.drop(df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].index)

# 3.删除NumberOfTime异常值
dfn98 = df.copy()
dfn98.loc[dfn98['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
dfn98.loc[dfn98['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
dfn98.loc[dfn98['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18

# 4.考虑之后，同时删除DebtRatio和删除RevolvingUtilizationOfUnsecuredLines 异常值
best_data = removed_debt_outliers.drop(removed_debt_outliers[removed_debt_outliers['RevolvingUtilizationOfUnsecuredLines'] > 10].index)

# 5.人为制造5%异常值
outlier_count = int(df.shape[0] * 0.05)
index = numpy.random.randint(0, df.shape[0], outlier_count)
add_outliers = df.copy()
for i in index:
    add_outliers.at[i, 'DebtRatio'] = numpy.random.randint(3489.025, 329664)

tester1 = Tester('SeriousDlqin2yrs')
tester3 = Tester('SeriousDlqin2yrs')

# tester.addDataset('process Missing', df)
# tester.addDataset('removed us', removed_debt_outliers) # 164 removed
# tester.addDataset('Removed 98s', dfn98) #269 removed
# tester.addDataset('Removed utilization', dfus) # 241 removed
# tester.addDataset('removed us', removed_debt_outliers) # 164 removed
# tester.addDataset('best_data', best_data)

paras = {}
n_estimators = [2, 4, 8, 16, 32, 64, 128, 150, 180, 200]
max_depth = [1, 2, 4, 6, 8, 10, 12, 32, 60, 100, 120, 150, 180, 200]
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
max_features = ['auto', 'sqrt', 'log2']
min_samples_leafs = np.linspace(1, 1500, 10, endpoint=True)
paras["n_estimators"] = n_estimators
paras["max_depth"] = max_depth
paras["max_features"] = max_features
paras["min_samples_splits"] = min_samples_splits
paras["min_samples_leafs"] = min_samples_leafs

to_tuning = 'min_samples_leafs'

rfc = []
for i in range(0, len(paras[to_tuning])):
    rfc.append(RandomForestClassifier(n_estimators=32, max_depth=8, random_state=0, max_features='auto',
                                      min_samples_leaf=int(min_samples_leafs[i]), oob_score=True))
    tester1.addModel(i, rfc[i])
    tester3.addModel(i, rfc[i])

tester1.addDataset('data', df)
test_auc, train_auc, oob_score, time_spent = tester1.runTests()
line1, = plt.plot(paras[to_tuning], train_auc, 'b', label='Train AUC')
line2, = plt.plot(paras[to_tuning], test_auc, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel(to_tuning)


# plt.subplot(312)
# line3 = plt.plot(paras[to_tuning], oob_score, 'y', label='oob score')
# plt.ylabel('oob score')
# plt.xlabel(to_tuning)
#
# plt.subplot(313)
# line4 = plt.plot(paras[to_tuning], time_spent, 'y', label='time_spent')
# plt.ylabel('time_spent')
# plt.xlabel(to_tuning)
# plt.show()


# A utility class to test all of our models on different datasets
# class Tester():
#     def __init__(self, target):
#         self.target = target
#         self.datasets = {}
#         self.models = {}
#         self.cache = {}  # we added a simple cache to speed things up
#
#     def addDataset(self, name, df):
#         self.datasets[name] = df.copy()
#
#     def addModel(self, name, model):
#         self.models[name] = model
#
#     def clearModels(self):
#         self.models = {}
#
#     def clearCache(self):
#         self.cache = {}
#
#     def testModelWithDataset(self, m_name, df_name, sample_len, cv):
#         if (m_name, df_name, sample_len, cv) in self.cache:
#             return self.cache[(m_name, df_name, sample_len, cv)]
#
#         clf = self.models[m_name]
#
#         if not sample_len:
#             sample = self.datasets[df_name]
#         else:
#             sample = self.datasets[df_name].sample(sample_len)
#
#         X = sample.drop([self.target], axis=1)
#         Y = sample[self.target]
#
#         s = cross_validate(clf, X, Y, scoring=['roc_auc'], cv=cv, n_jobs=-1)
#
#         if isinstance(clf, RandomForestClassifier):
#             clf.fit(X, Y)
#             s['oob'] = clf.oob_score_
#         self.cache[(m_name, df_name, sample_len, cv)] = s
#
#         return s
#
#     def runTests(self, sample_len=80000, cv=4):
#         # Tests the added models on all the added datasets
#         scores = {}
#         time_spent = []
#         for m_name in self.models:
#             for df_name in self.datasets:
#                 # print('Testing %s' % str((m_name, df_name)), end='')
#                 start = time.time()
#
#                 score = self.testModelWithDataset(m_name, df_name, sample_len, cv)
#                 scores[(m_name, df_name)] = score
#
#                 end = time.time()
#                 time_spent.append(end - start)
#                 # print(' -- %0.2fs ' % (end - start))
#
#         print('--- Top 10 Results ---')
#         for score in sorted(scores.items(), key=lambda x: -1 * x[1]['test_roc_auc'].mean())[::]:
#             auc = score[1]['test_roc_auc']
#             print("%s --> AUC: %0.4f (+/- %0.4f)" % (str(score[0]), auc.mean(), auc.std()), end='')
#             if 'oob' in score[1]:
#                 print("oob_score: %0.4f" % (score[1]['oob']))
#             else:
#                 print('')
#
#         test_auc = []
#         train_auc = []
#         oob_score = []
#         for score in scores.items():
#             test_auc.append(score[1]["test_roc_auc"].mean())
#             train_auc.append(score[1]["train_roc_auc"].mean())
#             oob_score.append(score[1]["oob"])
#         return test_auc, train_auc, oob_score, time_spent