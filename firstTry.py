import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.datasets import make_blobs

# 分别对三种数据源分类 three blocks, Two informative features+two clusters per class, Gaussian divided into four quantiles
# 分别用RF、GBDT分类
# 评估方法：1.画图对比  2.待选择
# 注意点：数据量？调参？如何定量评估？结合缺陷

# plt.figure(figsize=(2, 1))
# plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)


# three blocks

# sample 1
plt.subplot(331)
plt.title("Three blobs", fontsize='small')
X, y = make_blobs(n_samples=1000, n_features=2, centers=3)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

xxx, yyy = make_blobs(n_samples=1000, n_features=2, centers=3)
# RF
plt.subplot(332)
plt.title("Three blobs, RF", fontsize='small')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X, y)
yyy = clf.predict(xxx)
plt.scatter(xxx[:, 0], xxx[:, 1], marker='o', c=yyy)

# GBDT
plt.subplot(333)
plt.title("Three blobs, GBDT", fontsize='small')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
yyy = clf.predict(xxx)
plt.scatter(xxx[:, 0], xxx[:, 1], marker='o', c=yyy)

# sample 2
plt.subplot(334)
plt.title("Two informative features, two clusters per class", fontsize='small')
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

xxx, yyy = make_classification(n_samples=1000,n_features=2, n_redundant=0, n_informative=2)

# RF
plt.subplot(335)
plt.title("2, RF", fontsize='small')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X, y)
yyy = clf.predict(xxx)
plt.scatter(xxx[:, 0], xxx[:, 1], marker='o', c=yyy)

# GBDT
plt.subplot(336)
plt.title("2, GBDT", fontsize='small')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
yyy = clf.predict(xxx)
plt.scatter(xxx[:, 0], xxx[:, 1], marker='o', c=yyy)

# sample 3
plt.subplot(337)
plt.title("Gaussian", fontsize='small')

X, y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=4)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

xxx, yyy = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=4)

# RF
plt.subplot(338)
plt.title("Gaussian, RF", fontsize='small')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X, y)
yyy = clf.predict(xxx)
plt.scatter(xxx[:, 0], xxx[:, 1], marker='o', c=yyy)

# GBDT
plt.subplot(339)
plt.title("Gaussian, GBDT", fontsize='small')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
yyy = clf.predict(xxx)
plt.scatter(xxx[:, 0], xxx[:, 1], marker='o', c=yyy)


plt.show()

# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = RandomForestClassifier(n_estimators=10)
# print(clf.fit(X, Y))
#
# X, y = make_hastie_10_2(random_state=0)
# X_train, X_test = X[:2000], X[2000:]
# y_train, y_test = y[:2000], y[2000:]
#
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
# print(clf.score(X_test, y_test))
#
# X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
# X_train, X_test = X[:200], X[200:]
# y_train, y_test = y[:200], y[200:]
# est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
# print(mean_squared_error(y_test, est.predict(X_test)))
