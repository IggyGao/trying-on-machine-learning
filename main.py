import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from comparator import Comparator as Tester
pd.set_option('display.max_columns', None)


df = pd.read_csv(r"./data/processed_data.csv", engine="python")

tester1 = Tester('SeriousDlqin2yrs')
tester1.addDataset('processed_data', df)
# tester1.addModel('1', RandomForestClassifier(n_estimators=100, max_depth=16, max_features='auto', min_samples_leaf=100))
# tester1.addModel('2', RandomForestClassifier(n_estimators=100, max_depth=12, max_features='auto', min_samples_leaf=150))
# tester1.addModel('3', RandomForestClassifier(n_estimators=100, max_depth=8, max_features='auto'))

# tester1.addModel('1', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, subsample=0.95, max_depth=5, min_samples_leaf=43))
tester1.addModel('2', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, subsample=0.85, max_depth=5, min_samples_leaf=550))
tester1.runTests()

# n_estimators = [30, 50, 70, 90, 110, 130]
# learning_rate = [0.03,  0.05, 0.08,  0.1, 0.13, 0.15]
# plt.subplot(1, 2, 1)
# line1 = plt.plot(n_estimators, learning_rate, 'y', label='time_spent')
#
# plt.subplot(1, 2, 2)
# line2 = plt.plot(n_estimators, learning_rate, 'y', label='time_spent')
# plt.ylabel('time_spent')
#
# plt.ylim([0.84, 0.87])
#
# df = pd.read_csv(r"./data/processed_data.csv", engine="python")
# print(df.shape[0])
# # print("缺失值占比")
# print((df.shape[0] - df.count())/df.shape[0] *100)
#
# print(df.describe())
#
#
# df[df['DebtRatio'] > 3489.025][['SeriousDlqin2yrs', 'MonthlyIncome']].describe()

# df = df.drop(['Unnamed: 0'], axis=1)
# print(df[df['MonthlyIncome'].isnull().values==True].describe())

# print(df[df['DebtRatio'] >= 2382][['DebtRatio', 'MonthlyIncome']].describe())
