import numpy as np
import pandas as pd

# Calculate information value
def calc_iv(df, feature, target, pr=False):
    pd.set_option('display.max_columns', None)

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,  # Variable
                    val,  # Value
                    df[df[feature] == val].count()[feature],  # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]])  # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Good Rate'] = data['Good'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['Sub'] = (data['Distribution Good'] - data['Distribution Bad'])
    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data[['Value', 'Distribution Good', 'Distribution Bad', 'Sub' ,'WoE', 'IV']])
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()
    # print(iv)

    return iv, data


#导入数据
df = pd.read_csv(r"./data/cs-training.csv", engine="python")
df = df.dropna()

# 分组
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