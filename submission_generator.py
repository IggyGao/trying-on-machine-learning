import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#导入数据
train_data = pd.read_csv(r"./data/processed_data.csv", engine="python")
test_data = pd.read_csv(r"./data/cs-test.csv", engine="python")

X_train = train_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1)
Y_train = train_data['SeriousDlqin2yrs']

X_test = test_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1)
X_test.loc[(X_test.MonthlyIncome.isnull()), 'MonthlyIncome'] = X_test["MonthlyIncome"].median()
X_test.loc[(X_test.NumberOfDependents.isnull()), 'NumberOfDependents'] = X_test["NumberOfDependents"].median()

gbdt = RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=120)
gbdt.fit(X_train, Y_train)
Y_test = gbdt.predict_proba(X_test)

Y_test = pd.DataFrame(Y_test)
Y_test.columns = ['Probability', 'Probability2']
Y_test['Id'] = range(1, Y_test.shape[0]+1)
Y_test = Y_test.drop('Probability2', axis=1)
Y_test.to_csv(r"./data/submission.csv", index=False)
