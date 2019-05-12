import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv(r"./data/cs-training.csv", engine="python")
print("缺失值占比")
print((df.shape[0] - df.count())/df.shape[0] *100)
pd.set_option('display.max_columns', None)
print(df.describe())


