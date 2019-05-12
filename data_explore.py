import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from comparator import Comparator

#导入数据
df = pd.read_csv(r"./data/cs-training.csv", engine="python")

mid = df["MonthlyIncome"].median()
mode = df["MonthlyIncome"].mode().iloc[0]
# print(df["MonthlyIncome"].describe())
df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = mid

# 删除比较少的缺失值
df = df.dropna()

# 删除重复项
df = df.drop_duplicates()

# 1.删除DebtRatio 异常值
removed_debt_outliers = df.drop(df[df['DebtRatio'] > 3489.025].index)

# 2.删除RevolvingUtilizationOfUnsecuredLines 异常值
dfus = df.drop(df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].index)

# 3.替换NumberOfTime异常值
repalace98 = df.copy()
repalace98.loc[repalace98['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
repalace98.loc[repalace98['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
repalace98.loc[repalace98['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18

# 4.删除NumberOfTime异常值
drop98 = df.copy()
drop98 = drop98.drop(drop98[drop98['NumberOfTime30-59DaysPastDueNotWorse'] > 90].index)
drop98 = drop98.drop(drop98[drop98['NumberOfTime60-89DaysPastDueNotWorse'] > 90].index)
drop98 = drop98.drop(drop98[drop98['NumberOfTimes90DaysLate'] > 90].index)

# 5.人为制造5%异常值
add_outliers = df.copy()
outlier_count = int(df.shape[0] * 0.02)
index = numpy.random.randint(0, df.shape[0], outlier_count)
add_outliers.reset_index(drop=True, inplace=True)
for i in index:
    add_outliers.at[i, 'RevolvingUtilizationOfUnsecuredLines'] = numpy.random.randint(96, 98)

# 考虑之后，采用：删除RevolvingUtilizationOfUnsecuredLines 异常值，替换replaced 98s
best_data = repalace98.drop(repalace98[repalace98['RevolvingUtilizationOfUnsecuredLines'] > 10].index)
print(best_data.columns.values)
dfus = dfus.drop('Unnamed: 0', axis=1)
dfus.to_csv(r"./data/processed_data.csv")

tester = Comparator('SeriousDlqin2yrs')

tester.addDataset('missing data processed', df)
tester.addDataset('debt ratio outliers removed', removed_debt_outliers) # 164 removed
tester.addDataset('due outliers replaced', repalace98) #269 removed
tester.addDataset('utilization outliers removed、', dfus) # 241 removed
tester.addDataset('due outliers removed', drop98)
tester.addDataset('outliers added', add_outliers)
# tester.addDataset('best_data', best_data)

rfc = RandomForestClassifier(n_estimators=32, max_depth=8, random_state=0, max_features='auto', min_samples_leaf=1, oob_score=True)
dbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.07, max_depth=6)
tester.addModel('RF', rfc)
tester.addModel('GBDT', dbdt)

test_auc, train_auc, time_spent = tester.runTests()



