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

# states={"ID" : "用户ID",
#         "SeriousDlqin2yrs":"好坏客户",
#         "RevolvingUtilizationOfUnsecuredLines":"可用额度比值",
#         "age":"年龄",
#         "NumberOfTime30-59DaysPastDueNotWorse":"逾期30-59天笔数",
#         "DebtRatio":"负债率",
#         "MonthlyIncome":"月收入",
#         "NumberOfOpenCreditLinesAndLoans":"信贷数量",
#         "NumberOfTimes90DaysLate":"逾期90天笔数",
#         "NumberRealEstateLoansOrLines":"固定资产贷款量",
#         "NumberOfTime60-89DaysPastDueNotWorse":"逾期60-89天笔数",
#         "NumberOfDependents":"家属数量"}
# df.rename(columns=states, inplace=True)

# df.head(5)


# 数据清洗

# # 1.缺失值
# print("缺失值占比")
# print((df.shape[0] - df.count())/df.shape[0] *100)
#
# #用众数填充较多的缺失值
# df=df.fillna({"月收入":df["月收入"].mode()})
# df.info()
#
# #少量缺失值直接删除
df1=df.dropna()


# 2.异常值处理

# print(df1[df1["好坏客户"] not in ["Y", "N"]].count())
# print("RevolvingUtilizationOfUnsecuredLines异常：" + str(df[df["RevolvingUtilizationOfUnsecuredLines"] > 1].shape[0]/df.shape[0] *100))
# print("age异常：" + str((df[df["age"] <= 0].shape[0] + df[df["age"] > 120].shape[0])/df.shape[0] *100))
# print("NumberOfTime30-59DaysPastDueNotWorse异常：" + str(df[df["NumberOfTime30-59DaysPastDueNotWorse"] < 0].shape[0]/df.shape[0] *100))
# print("DebtRatio：" + str(df[df["DebtRatio"] < 0].shape[0]/df.shape[0] *100))
# print("MonthlyIncome异常：" + str(df[df["MonthlyIncome"] < 0].shape[0]/df.shape[0] *100))
# print("信贷数量异常：" + str(df[df["信贷数量"] < 0].shape[0]/df.shape[0] *100))
# print("逾期90天笔数异常：" + str(df[df["逾期90天笔数"] < 0].shape[0]/df.shape[0] *100))
# print("固定资产贷款量异常：" + str(df[df["固定资产贷款量"] < 0].shape[0]/df.shape[0] *100))
# print("逾期60-89天笔数异常："+ str(df[df["逾期60-89天笔数"] < 0].shape[0]/df.shape[0] *100))
# print("家属数量异常：" + str(df[df["家属数量"] < 0].shape[0]/df.shape[0] *100))

# # 直接删除年龄和可用额度比值异常
# df1=df1[df1["可用额度比值"]<=1]
# df1=df1[df1["年龄"]>0]
# df1=df1[df1["年龄"]<120]

# 3.离群点删除

x1=df1["RevolvingUtilizationOfUnsecuredLines"]
x2=df1["age"]
x2=df1["NumberOfTime30-59DaysPastDueNotWorse"]
x3=df1["DebtRatio"]
x4=df1["MonthlyIncome"]
x5=df1["NumberOfOpenCreditLinesAndLoans"]
x6=df1["NumberOfTimes90DaysLate"]
x7=df1["NumberRealEstateLoansOrLines"]
x8=df1["NumberOfTime60-89DaysPastDueNotWorse"]
x9=df1["NumberOfDependents"]
fig=plt.figure(1)

plt.subplot(251)
plt.title('RevolvingUtilizationOfUnsecuredLines')
plt.boxplot([df1["RevolvingUtilizationOfUnsecuredLines"]])

plt.subplot(252)
plt.title("age")
plt.boxplot([df1["age"]])

plt.subplot(253)
plt.title("NumberOfTime\n30-59DaysPastDueNotWorse")
plt.boxplot([df1["NumberOfTime30-59DaysPastDueNotWorse"]])

plt.subplot(254)
plt.title("DebtRatio")
plt.boxplot([df1["DebtRatio"]])

plt.subplot(255)
plt.title("MonthlyIncome")
plt.boxplot([df1["MonthlyIncome"]])

plt.subplot(256)
plt.title("NumberOf\nOpenCreditLinesAndLoans")
plt.boxplot([df1["NumberOfOpenCreditLinesAndLoans"]])

plt.subplot(257)
plt.title("NumberOfTimes\n90DaysLate")
plt.boxplot([df1["NumberOfTimes90DaysLate"]])

plt.subplot(258)
plt.title("NumberRealEstateLoansOrLines")
plt.boxplot([df1["NumberRealEstateLoansOrLines"]])

plt.subplot(259)
plt.title("NumberOfTime\n60-89DaysPastDueNotWorse")
plt.boxplot([df1["NumberOfTime60-89DaysPastDueNotWorse"]])

plt.subplot(2, 5, 10)
plt.title("NumberOfDependents")
plt.boxplot([df1["NumberOfDependents"]])


# 以下为示例中删除的部分
# df1=df1[df1["逾期30-59天笔数"]<80]
# df1=df1[df1["固定资产贷款量"]<50]

#  以下是我认为还应该删除部分？？？
# df1 = df1[df1["逾期90天笔数"]<80]
# # df1=df1[df1["固定资产贷款量"]<50]

plt.show()





