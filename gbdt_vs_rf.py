import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from comparator import Comparator
import numpy

df = pd.read_csv(r"./data/processed_data.csv", engine="python")

add_outliers = df.copy()
outlier_count = int(df.shape[0] * 0.05)
index = numpy.random.randint(0, df.shape[0], outlier_count)
add_outliers.reset_index(drop=True, inplace=True)
for i in index:
    add_outliers.at[i, 'DebtRatio'] = numpy.random.randint(3000, 30000)


comparator = Comparator('SeriousDlqin2yrs')

comparator.addDataset('data', df)
comparator.addDataset('outliers added', add_outliers)

# comparator.addModel('tuned RF', RandomForestClassifier(n_estimators=100, max_depth=16, max_features='auto', min_samples_leaf=100))
# comparator.addModel('default RF', RandomForestClassifier())
comparator.addModel('tuned GBDT', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, subsample=0.85, max_depth=5, min_samples_leaf=500))
comparator.addModel('default GBDT', GradientBoostingClassifier())

comparator.runTests()
