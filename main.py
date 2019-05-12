import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# df = pd.read_csv(r"./data/cs-training.csv", engine="python")
# print(df['NumberOfTimes90DaysLate'].describe())

df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]], index=[4, 5, 6], columns=['A', 'B', 'C'])
df.iloc[:, 0]+1
df.at[4, 'B']
