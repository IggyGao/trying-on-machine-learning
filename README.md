# trying-on-machine-learning报告

## 综述：
尝试探索kaggle网站上信用卡用户分类的competition([地址链接](https://www.kaggle.com/c/GiveMeSomeCredit/overview))，进行数据探索-数据清洗-模型训练-模型评估等一系列操作。内容主要包括：


#### 1. 数据探索：
- 处理空值。分析空值占比，占比极少的直接删除，较多的进行填充。
- 处理离群点。通过箱型图+knn找出可能的异常值。

对第二步中找出的异常值分别处理，处理后的数据为数据集1-5。使用模型分别训练数据集1-5，对比评估，从而选出合适的数据处理方式，用于接下来的模型训练。

#### 2. 模型训练：
- 分别对随机森林和GBDT进行调参，考虑ROC-AOC、速度、是否过拟合，从而找出较好的参数。
- 从预测准确性、抗噪声能力等方面对两种模型进行对比。

#### 3. 结果：
将测试数据分测试集和验证集，交叉验证AUC-ROC。较好的结果如下（具体参数见文章第二部分）：

| 模型 | mean | std|
| ------ | ------ | ------ |
| RF | 0.8644 | 0.0015 |
| GBDT | 0.8632 | 0.0011 |

使用对应模型预测测试集，生成submission.csv提交，网站评分如下图

![avatar](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/my_score.png?raw=true)


文件列表

    comparator.py  ------------ 对比器，用于生成各模型/数据集的ROC列表和调参对比图
    data_explore.py  ---------- 数据清洗
    rf_tuning.py  ------------- 随机森林调参 
    gbdt_tuning.py  ----------- GBDT调参  
   

## 具体内容

### 一、数据探索
1. 空值

|  列名     |缺失值占比|
| ------ | ------ | 
|SeriousDlqin2yrs                        | 0.000000
|RevolvingUtilizationOfUnsecuredLines     |0.000000
|age                                      |0.000000
|NumberOfTime30-59DaysPastDueNotWorse     |0.000000
|DebtRatio                                |0.000000
|MonthlyIncome                           |19.820667
|NumberOfOpenCreditLinesAndLoans         |0.000000
|NumberOfTimes90DaysLate                  |0.000000
|NumberRealEstateLoansOrLines             |0.000000
|NumberOfTime60-89DaysPastDueNotWorse     |0.000000
|NumberOfDependents                       |2.616000

2. 离群点

![avatar](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/overview.png?raw=true)

|           |  SeriousDlqin2yrs | RevolvingUtilizationOfUnsecuredLines  |
| ------ | ------ | ------ | 
|count  |150000.000000     |150000.000000                         |150000.000000 |  
|mean    |75000.500000      |    0.066840                          |    6.048438  | 
|std     |43301.414527       |   0.249746                           | 249.755371   |
|min     |    1.000000        |  0.000000                            |  0.000000   |
|25%     |37500.750000         | 0.000000                             | 0.029867   |
|50%     |75000.500000         | 0.000000                             | 0.154181   |
|75%    |112500.250000         | 0.000000                             | 0.559046   |
|max    |150000.000000          |1.000000                          |50708.000000   |



3. 引入模型评估
- 评估处理方法
- 评估数据质量

### 二、模型探索
1. Rondom Forest调参 
2. GBDT调参
3. 模型对比

- 对比auc-roc（为什么用这个）
- 加入离群点对比抗噪声能力

4. 关于roc的思考。阈值探索。

