# trying-on-machine-learning

目录?

## 综述：
尝试探索kaggle网站上信用卡用户分类的competition([详情链接](https://www.kaggle.com/c/GiveMeSomeCredit/overview))，进行数据探索-数据清洗-模型训练-模型评估等一系列操作。内容主要包括：


#### 1. 数据探索：
- 空值。

    分析空值占比，占比极少的直接删除，较多的进行填充。
    
- 离群点

    找出可能的异常值，分析情况，对应不同处理方式（删除/替换）生成数据集1-5。使用模型训练数据集1-5，对比评估，从而选出合适的数据处理方式，用于接下来的模型训练。

#### 2. 模型训练和对比分析

对比不同参数下随机森林（以下简称RF）和GBDT模型的好坏，包括：分类的正确性（AUC_ROC）、速度、是否过拟合等，寻找到此数据集下两种模型较为合适的参数。涉及参数如下：

| | RF参数 | GBDT参数 | 
|-------| ------ | ------ | 
| Bagging相关参数|n_estimators<br>max_features|n_estimators和learning_rate<br>subsample<br>loss|
| 学习器相关参数|max_depth<br>min_sample_leaf<br>min_samples_split|同随机森林|

然后从以下反面对两个模型进行对比分析：
    
   - 抗噪能力：RF > GBDT
   - 速度: 训练速度RF > GBDT; 分类速度DBDT>RF
   - 分类准确度: GBDT 整体略好
   - 调参: RF更简单

#### 3. 结果：
将测试数据分测试集和验证集，交叉验证AUC-ROC。较好的结果如下（具体参数见文章第二部分）：

| 模型 | mean | std|
| ------ | ------ | ------ |
| RF | 0.8644 | 0.0015 |
| GBDT | 0.8632 | 0.0011 |

使用对应模型对测试集进行分类，生成submission.csv提交，网站评分如下图

<img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/my_score.png?raw=true" width="60%" >

#### 4. 延伸思考

#### 5. 文件列表

    comparator.py  ------------ 对比器，用于生成各模型/数据集的ROC列表和调参对比图
    data_explore.py  ---------- 数据清洗
    rf_tuning.py  ------------- 随机森林调参 
    gbdt_tuning.py  ----------- GBDT调参  
   
## 具体处理和数据支撑

### 一、数据探索
1. 数据描述

    共11维数据，describe看一下大致情况。

|           |  SeriousDlqin2yrs | Revolving<br>Utilization<br>OfUnsecuredLines  |age | NumberOfTime<br>30-59Days<br>PastDueNotWorse    |  DebtRatio | MonthlyIncome | NumberOfOpen<br>CreditLines<br>AndLoans|NumberOfTimes<br>90DaysLate  |NumberRealEstate<br>LoansOrLines|NumberOfTime<br>60-89Days<br>PastDueNotWorse | NumberOfDependents  |
| ------ | ------ | ------ | ----- | ---------| -----------|------|-------|------|--------|----|-----|
|count  |150000.000000  |150000.000000 |  150000.000000     |  150000.000000 | 150000.000000  |  1.202690e+05     |        150000.000000  | 150000.000000              |   150000.000000 | 150000.000000  |     146076.000000 | 
|mean  |    0.066840 |    6.048438  |  52.295207     |    0.421033    | 353.005076   |   6.670221e+03        |                 8.452760   |0.265973              |        1.018240  |  0.240387     |       0.757222  |
|std      |   0.249746   | 249.755371   |14.771866   |   4.192781   | 2037.818523  |    1.438467e+04     |                    5.145951   |  4.169304              |        1.129771   | 4.155179     |       1.115086  |
|min  |  0.000000   |  0.000000   | 0.000000        |       0.000000      | 0.000000   |    0.000000e+00         |                0.000000  | 0.000000            |          0.000000 |    0.000000  |          0.000000  |
|25%     | 0.000000    | 0.029867   |  41.000000    |     0.000000     |  0.175074   |3.400000e+03              |           5.000000 | 0.000000               |       0.000000   |0.000000    |        0.000000  |
|50%     | 0.000000     | 0.154181   | 52.000000     |          0.000000    |   0.366508    5.400000e+03     |                    8.000000 | 0.000000          |            1.000000 | 0.000000 |           0.000000  |
|75%   | 0.000000   | 0.559046   |63.000000          |    0.000000  |     0.868254   |  8.249000e+03  |                      11.000000  |0.000000            |          2.000000   | 0.000000    |        1.000000  |
|max    |1.000000    |50708.000000   |109.000000       |  98.000000  |329664.000000 |  3.008750e+06         |               58.000000 | 98.000000            |         54.000000|    98.000000   |        20.000000  |



2. 空值

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

3. 引入模型评估
- 评估处理方法
- 评估数据质量
- 评估importance

### 二、模型探索

1. Rondom Forest调参 
    - n_estimators
    - max_features
    - max_depth、min_sample_leaf、min_samples_split
    
2. GBDT调参

    a. boosting参数
    - n_estimators和learning_rate
    - subsample
    - loss
    
    b. 弱学习器参数（同RF）
  
    
3. 模型对比

- RF抗噪能力更强（加入离群点对比抗噪声能力）
- RF训练速度更快，但是分类速度慢
- RF调参相对简单
- GBDT正确率更高

## 其他一些思考

1. 两者思路上的不同

    GBDT：shallow trees （ weak learners，high bias, low variance） +  gradient 提高bias

    RF：fully grown trees（low bias, high variance） + bagging 提高variance

2. 本文以外的思考

3. 关于正确率评判方式的思考

4. 阈值探索

