import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', None)


df = pd.read_csv(r"./data/cs-training.csv", engine="python")
# print("缺失值占比")
# print((df.shape[0] - df.count())/df.shape[0] *100)
#
# print(df.describe())
#
#
# df[df['DebtRatio'] > 3489.025][['SeriousDlqin2yrs', 'MonthlyIncome']].describe()

df= df.drop(['Unnamed: 0'], axis=1)
# print(df[df['MonthlyIncome'].isnull().values==True].describe())

print(df[df['RevolvingUtilizationOfUnsecuredLines'] > 1][['RevolvingUtilizationOfUnsecuredLines', 'SeriousDlqin2yrs']].describe())

print(df[df['DebtRatio'] >= 500][['RevolvingUtilizationOfUnsecuredLines', 'MonthlyIncome']].describe())

print(df[df['MonthlyIncome'] < 2][['RevolvingUtilizationOfUnsecuredLines', 'MonthlyIncome']].describe())