#导入相关库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示坐标轴负号

#导入数据
df = pd.read_csv(r"./data/cs-training.csv", engine="python")
df.head(5)
print(df.columns)

states={"ID" : "用户ID",
        "SeriousDlqin2yrs":"好坏客户",
        "RevolvingUtilizationOfUnsecuredLines":"可用额度比值",
        "age":"年龄",
        "NumberOfTime30-59DaysPastDueNotWorse":"逾期30-59天笔数",
        "DebtRatio":"负债率",
        "MonthlyIncome":"月收入",
        "NumberOfOpenCreditLinesAndLoans":"信贷数量",
        "NumberOfTimes90DaysLate":"逾期90天笔数",
        "NumberRealEstateLoansOrLines":"固定资产贷款量",
        "NumberOfTime60-89DaysPastDueNotWorse":"逾期60-89天笔数",
        "NumberOfDependents":"家属数量"}
df.rename(columns=states, inplace=True)

df.head(5)


# 数据清洗

# 1.缺失值
print("缺失值占比")
print((df.shape[0] - df.count())/df.shape[0] *100)

#用众数填充较多的缺失值
df=df.fillna({"月收入":df["月收入"].mode()})
df.info()

#少量缺失值直接删除
df1=df.dropna()


# 2.异常值处理

# print(df1[df1["好坏客户"] not in ["Y", "N"]].count())
print("可用额度比值异常：" + str(df[df["可用额度比值"] > 1].shape[0]/df.shape[0] *100))
print("年龄异常：" + str((df[df["年龄"] <= 0].shape[0] + df[df["年龄"] > 120].shape[0])/df.shape[0] *100))
print("逾期30-59天笔数异常：" + str(df[df["逾期30-59天笔数"] < 0].shape[0]/df.shape[0] *100))
print("负债率异常：" + str(df[df["负债率"] < 0].shape[0]/df.shape[0] *100))
print("月收入 异常：" + str(df[df["月收入"] < 0].shape[0]/df.shape[0] *100))
print("信贷数量异常：" + str(df[df["信贷数量"] < 0].shape[0]/df.shape[0] *100))
print("逾期90天笔数异常：" + str(df[df["逾期90天笔数"] < 0].shape[0]/df.shape[0] *100))
print("固定资产贷款量异常：" + str(df[df["固定资产贷款量"] < 0].shape[0]/df.shape[0] *100))
print("逾期60-89天笔数异常："+ str(df[df["逾期60-89天笔数"] < 0].shape[0]/df.shape[0] *100))
print("家属数量异常：" + str(df[df["家属数量"] < 0].shape[0]/df.shape[0] *100))

# 直接删除年龄和可用额度比值异常
df1=df1[df1["可用额度比值"]<=1]
df1=df1[df1["年龄"]>0]
df1=df1[df1["年龄"]<120]

# 3.离群点删除

x1=df1["可用额度比值"]
x2=df1["年龄"]
x2=df1["逾期30-59天笔数"]
x3=df1["负债率"]
x4=df1["月收入"]
x5=df1["信贷数量"]
x6=df1["逾期90天笔数"]
x7=df1["固定资产贷款量"]
x8=df1["逾期60-89天笔数"]
x9=df1["家属数量"]
fig=plt.figure(1)

plt.subplot(251)
plt.title(u'可用额度比值')
plt.boxplot([df1["可用额度比值"]])

plt.subplot(252)
plt.title("nial")
plt.boxplot([df1["年龄"]])

plt.subplot(253)
plt.title("逾期30-59天笔数")
plt.boxplot([df1["逾期30-59天笔数"]])

plt.subplot(254)
plt.title("负债率")
plt.boxplot([df1["负债率"]])

plt.subplot(255)
plt.title("月收入")
plt.boxplot([df1["月收入"]])

plt.subplot(256)
plt.title("信贷数量")
plt.boxplot([df1["信贷数量"]])

plt.subplot(257)
plt.title("逾期90天笔数")
plt.boxplot([df1["逾期90天笔数"]])

plt.subplot(258)
plt.title("固定资产贷款量")
plt.boxplot([df1["固定资产贷款量"]])

plt.subplot(259)
plt.title("逾期60-89天笔数")
plt.boxplot([df1["逾期60-89天笔数"]])

plt.subplot(2, 5, 10)
plt.title("家属数量")
plt.boxplot([df1["家属数量"]])


# 以下为示例中删除的部分
# df1=df1[df1["逾期30-59天笔数"]<80]
# df1=df1[df1["固定资产贷款量"]<50]

#  以下是我认为还应该删除的部分？？？
df1 = df1[df1["逾期90天笔数"]<80]
# # df1=df1[df1["固定资产贷款量"]<50]

#画了那些图来熟悉数据
#预期结论有？


plt.show()



