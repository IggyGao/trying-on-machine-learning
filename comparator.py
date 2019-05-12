import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
import numpy
import time

# A utility class to test all of our models on different datasets
class Comparator():
    def __init__(self, target):
        self.target = target
        self.datasets = {}
        self.models = {}
        self.cache = {}  # we added a simple cache to speed things up

    def addDataset(self, name, df):
        self.datasets[name] = df.copy()

    def addModel(self, name, model):
        self.models[name] = model

    def clearModels(self):
        self.models = {}

    def clearCache(self):
        self.cache = {}

    def testModelWithDataset(self, m_name, df_name, sample_len, cv):
        if (m_name, df_name, sample_len, cv) in self.cache:
            return self.cache[(m_name, df_name, sample_len, cv)]

        clf = self.models[m_name]

        if not sample_len:
            sample = self.datasets[df_name]
        else:
            sample = self.datasets[df_name].sample(sample_len)

        # X = sample.drop([self.target, 'Unnamed: 0'], axis=1)
        X = sample.drop([self.target], axis=1)
        Y = sample[self.target]

        s = cross_validate(clf, X, Y, scoring=['roc_auc'], cv=cv, n_jobs=-1)

        # if isinstance(clf, RandomForestClassifier):
        #     clf.fit(X, Y)
        #     s['oob'] = clf.oob_score_
        # self.cache[(m_name, df_name, sample_len, cv)] = s

        return s

    def runTests(self, sample_len=80000, cv=4):
        # Tests the added models on all the added datasets
        scores = {}
        time_spent = []
        for m_name in self.models:
            for df_name in self.datasets:
                # print('Testing %s' % str((m_name, df_name)), end='')
                start = time.time()

                score = self.testModelWithDataset(m_name, df_name, sample_len, cv)
                scores[(m_name, df_name)] = score

                end = time.time()
                time_spent.append(end - start)
                # print(' -- %0.2fs ' % (end - start))

        print('--- Top 10 Results ---')
        for score in sorted(scores.items(), key=lambda x: -1 * x[1]['test_roc_auc'].mean())[::]:
            auc = score[1]['test_roc_auc']
            print("%s --> AUC: %0.4f (+/- %0.4f)" % (str(score[0]), auc.mean(), auc.std()))
            # if 'oob' in score[1]:
            #     print("oob_score: %0.4f" % (score[1]['oob']))
            # else:
            #     print('')

        test_auc = []
        train_auc = []
        # oob_score = []
        for score in scores.items():
            test_auc.append(score[1]["test_roc_auc"].mean())
            train_auc.append(score[1]["train_roc_auc"].mean())
            # oob_score.append(score[1]["oob"])
        return test_auc, train_auc, time_spent
