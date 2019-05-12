import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]], index=[4, 5, 6], columns=['A', 'B', 'C'])
df.at[4, 'B'] = 10
df.at[4, 'B']