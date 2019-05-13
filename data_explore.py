import pandas as pd
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from comparator import Comparator

#导入数据
df = pd.read_csv(r"./data/cs-training.csv", engine="python")

# 用中位数填充MonthlyIncome的空值
mid = df["MonthlyIncome"].median()
df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = mid

# 删除比较少的缺失值
df = df.dropna()

# 删除重复项
df = df.drop_duplicates()

# 1.删除DebtRatio 异常值
removed_debt_outliers1 = df.drop(df[df['DebtRatio'] > 3489.025].index)

# 1.替换DebtRatio和MonthlyIncome的异常值
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



