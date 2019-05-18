import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)


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
df = pd.read_csv(r"./data/cs-training.csv", engine="python")
# # print("缺失值占比")
# print((df.shape[0] - df.count())/df.shape[0] *100)
#
# print(df.describe())
#
#
# df[df['DebtRatio'] > 3489.025][['SeriousDlqin2yrs', 'MonthlyIncome']].describe()

# df = df.drop(['Unnamed: 0'], axis=1)
# print(df[df['MonthlyIncome'].isnull().values==True].describe())

print(df[df['DebtRatio'] >= 2382][['DebtRatio', 'MonthlyIncome']].describe())
