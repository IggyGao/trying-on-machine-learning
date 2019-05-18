import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import time

#导入数据
train_data = pd.read_csv(r"./data/processed_data.csv", engine="python")
X_train = train_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1)
Y_train = train_data['SeriousDlqin2yrs']

test_data = pd.read_csv(r"./data/cs-test.csv", engine="python")
# test_data = test_data.dropna()
X_test = test_data.drop(['SeriousDlqin2yrs', 'Unnamed: 0', 'MonthlyIncome'], axis=1)
X_test.loc[(X_test.NumberOfDependents.isnull()), 'NumberOfDependents'] = X_test["NumberOfDependents"].median()

rf = RandomForestClassifier(n_estimators=100, max_depth=16, max_features='auto', min_samples_leaf=100)
start = time.time()
rf.fit(X_train, Y_train)
end1 = time.time()
Y_test = rf.predict_proba(X_test)
end2 = time.time()
print(' RF train costs -- %0.2fs ' % (end1 - start))
print(' RF test costs -- %0.2fs ' % (end2 - end1))

Y_test = pd.DataFrame(Y_test)
Y_test.columns = ['Probability', 'Probability2']
Y_test['Id'] = range(1, Y_test.shape[0]+1)
Y_test = Y_test.drop('Probability2', axis=1)
Y_test.to_csv(r"./data/rf.csv", index=False)

gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, subsample=0.85, max_depth=5, min_samples_leaf=550)
start = time.time()
gbdt.fit(X_train, Y_train)
end1 = time.time()
Y_test = gbdt.predict_proba(X_test)
end2 = time.time()
print('GBDT train costs -- %0.2fs ' % (end1 - start))
print('GBDT test costs -- %0.2fs ' % (end2 - end1))

Y_test = pd.DataFrame(Y_test)
Y_test.columns = ['Probability', 'Probability2']
Y_test['Id'] = range(1, Y_test.shape[0]+1)
Y_test = Y_test.drop('Probability2', axis=1)
Y_test.to_csv(r"./data/gbdt.csv", index=False)
