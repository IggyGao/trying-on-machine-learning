import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from comparator import Comparator

pd.set_option('display.max_columns', None)

#导入数据
df = pd.read_csv(r"./data/cs-training.csv", engine="python")

print(df["MonthlyIncome"].describe())

mid = df["MonthlyIncome"].median()
df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = mid

# df = df.dropna()

all = df.shape[0]
print(df[(df['MonthlyIncome'] >= 0) & (df['MonthlyIncome'] < 2000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 2000) & (df['MonthlyIncome'] < 3000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 3000) & (df['MonthlyIncome'] < 4000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 4000) & (df['MonthlyIncome'] < 5000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 5000) & (df['MonthlyIncome'] < 6000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 6000) & (df['MonthlyIncome'] < 7000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 7000) & (df['MonthlyIncome'] < 9000)].shape[0]/all * 100)
print(df[(df['MonthlyIncome'] >= 9000) & (df['MonthlyIncome'] < 12000)].shape[0]/all * 100)
print(df[df['MonthlyIncome'] >= 12000].shape[0]/all*100)


income_iv = df.copy()

income_iv.loc[(df['MonthlyIncome'] >= 0) & (df['MonthlyIncome'] < 2000), 'MonthlyIncome'] = 0
income_iv.loc[(df['MonthlyIncome'] >= 2000) & (df['MonthlyIncome'] < 3000), 'MonthlyIncome'] = 2000
income_iv.loc[(df['MonthlyIncome'] >= 3000) & (df['MonthlyIncome'] < 4000), 'MonthlyIncome'] = 3000
income_iv.loc[(df['MonthlyIncome'] >= 4000) & (df['MonthlyIncome'] < 5000), 'MonthlyIncome'] = 4000
income_iv.loc[(df['MonthlyIncome'] >= 5000) & (df['MonthlyIncome'] < 6000), 'MonthlyIncome'] = 5000
income_iv.loc[(df['MonthlyIncome'] >= 6000) & (df['MonthlyIncome'] < 7000), 'MonthlyIncome'] = 6000
income_iv.loc[(df['MonthlyIncome'] >= 7000) & (df['MonthlyIncome'] < 9000), 'MonthlyIncome'] = 7000
income_iv.loc[(df['MonthlyIncome'] >= 9000) & (df['MonthlyIncome'] < 12000), 'MonthlyIncome'] = 9000
income_iv.loc[df['MonthlyIncome'] >= 12000, 'MonthlyIncome'] = 12000


iv, data = calc_iv(income_iv, 'MonthlyIncome', 'SeriousDlqin2yrs', pr=True)


# 缺失量接近20%，考虑直接删除MonthlyIncome维度
drop_income = df.drop(["MonthlyIncome"], axis=1)

# 用中位数填充MonthlyIncome的空值
mid = df["MonthlyIncome"].median()
df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = mid

# 删除比较少的缺失值
df = df.dropna()
drop_income = drop_income.dropna()

# 删除重复项
df = df.drop_duplicates()
drop_income.dropna()

# 1.删除DebtRatio异常值
removed_debt_outliers1 = df.drop(df[df['DebtRatio'] > 3489.025].index)

# 1.99分位点替换DebtRatio和MonthlyIncome的异常值
repalace_income = df.copy()
repalace_income.loc[repalace_income['MonthlyIncome'] <= 1, 'MonthlyIncome'] = mid
debt_mid = df["DebtRatio"].median()
repalace_income.loc[repalace_income['DebtRatio'] <= 500, 'DebtRatio'] = debt_mid

# 1.删除DebtRatio 异常值
removed_debt_outliers2 = df.drop(df[df['DebtRatio'] > 500].index)

# 2.删除RevolvingUtilizationOfUnsecuredLines 异常值
dfus = df.drop(df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].index)

# 3.替换逾期异常值
repalace98 = df.copy()
repalace98.loc[repalace98['NumberOfTime30-59DaysPastDueNotWorse'] > 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 18
repalace98.loc[repalace98['NumberOfTime60-89DaysPastDueNotWorse'] > 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 18
repalace98.loc[repalace98['NumberOfTimes90DaysLate'] > 90, 'NumberOfTimes90DaysLate'] = 18

# 4.删除逾期异常值
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
# best_data = repalace98.drop(repalace98[repalace98['RevolvingUtilizationOfUnsecuredLines'] > 10].index)
# print(best_data.columns.values)
df = df.drop('Unnamed: 0', axis=1)
df.to_csv(r"./data/processed_data.csv")

tester = Comparator('SeriousDlqin2yrs')

tester.addDataset('missing data processed', df)
tester.addDataset('debt ratio outliers 1 removed', removed_debt_outliers1) # 164 removed
tester.addDataset('debt ratio outliers 2 removed', removed_debt_outliers2) # 164 removed
tester.addDataset('debt ratio outliers replaced', repalace_income) # 164 removed
tester.addDataset('overdue outliers replaced', repalace98) #269 removed
tester.addDataset('utilization outliers removed', dfus) # 241 removed
tester.addDataset('overdue outliers removed', drop98)
# tester.addDataset('outliers added', add_outliers)
# tester.addDataset('best_data', best_data)

rf_default = RandomForestClassifier()
dbdt_default = GradientBoostingClassifier()
tester.addModel('default RF', rf_default)
tester.addModel('default GBDT ', dbdt_default)

rf = RandomForestClassifier(n_estimators=32, max_depth=8, random_state=0, max_features='auto', oob_score=True)
dbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
tester.addModel('RF', rf)
tester.addModel('GBDT', dbdt)

tester.addModel('Simple SVM', svm.LinearSVC())

test_auc, train_auc, time_spent = tester.runTests()



