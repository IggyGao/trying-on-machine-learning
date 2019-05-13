# trying-on-machine-learning

## 综述：
尝试探索kaggle网站上信用卡用户分类的competition([详情链接](https://www.kaggle.com/c/GiveMeSomeCredit/overview))，进行数据探索-数据清洗-模型训练-模型评估等一系列操作，内容主要包括：


#### 1. 数据探索：
    
   主要是离群点的处理方式。
   找出可能的异常值，分析情况，对应不同处理方式（删除/替换）生成多种数据集。使用粗略调参的模型分别训练数据集，对比评估，从而选出合适的数据处理方式，用于接下来的模型训练。

#### 2. 模型训练和对比分析

对比不同参数下随机森林（以下简称RF）和GBDT模型的好坏，包括：分类的正确性（AUC_ROC）、速度、是否过拟合等，寻找到此数据集下两种模型较为合适的参数。涉及参数如下：

| | RF参数 | GBDT参数 | 
|-------| ------ | ------ | 
| Bagging/Boosting相关参数|n_estimators<br>max_features|n_estimators和learning_rate<br>subsample<br>loss|
| 学习器相关参数|max_depth<br>min_sample_leaf<br>min_samples_split|同RF|

RF核心思想：fully grown trees（低bias高variance） + bagging （降低variance）。

GBDT核心思想：shallow trees（高bias低variance） + gradient （降低bias）

所以如果分类偏差较大，RF应该调节学习器相关参数，GBDT应该调节Boosting相关参数。

如果抗噪能力比较弱，RF应该调节Bagging参数，GBDT应该调节学习器相关参数。

然后从以下反面对两个模型进行对比分析：
    
   - 抗噪能力：RF > GBDT
   - 速度: 训练速度RF > GBDT; 分类速度DBDT>RF
   - 分类准确度: GBDT 整体略好
   - 调参: RF更简单

#### 3. 结果：
将测试数据分测试集和验证集，交叉验证算出AUC-ROC的平均值。较好的结果如下（具体参数见文章第二部分）：

| 模型 | mean | std|
| ------ | ------ | ------ |
| RF | 0.8644 | 0.0025 |
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

#### 1. 数据描述

    共11维数据，describe看一下大致情况。加分析。

|           |  SeriousDlqin2yrs | Revolving<br>Utilization<br>OfUnsecuredLines  |age | NumberOfTime<br>30-59Days<br>PastDueNotWorse    |  DebtRatio | MonthlyIncome | NumberOfOpen<br>CreditLines<br>AndLoans|NumberOfTimes<br>90DaysLate  |NumberRealEstate<br>LoansOrLines|NumberOfTime<br>60-89Days<br>PastDueNotWorse | NumberOfDependents  |
| ------ | ------ | ------ | ----- | ---------| -----------|------|-------|------|--------|----|-----|
|count  |150000.000000  |150000.000000 |  150000.000000     |  150000.000000 | 150000.000000  |  1.202690e+05 |150000.000000  | 150000.000000 |   150000.000000 | 150000.000000  |     146076.000000 | 
|mean  |    0.066840 |    6.048438  |  52.295207     |    0.421033    | 353.005076   |   6.670221e+03        | 8.452760   |0.265973      |  1.018240  |  0.240387     |       0.757222  |
|std      |   0.249746   | 249.755371   |14.771866   |   4.192781   | 2037.818523  |    1.438467e+04     |  5.145951   |  4.169304   | 1.129771   | 4.155179     |       1.115086  |
|min  |  0.000000   |  0.000000   | 0.000000        |       0.000000      | 0.000000   |    0.000000e+00         |  0.000000  | 0.000000 |   0.000000 |    0.000000  |          0.000000  |
|25%     | 0.000000    | 0.029867   |  41.000000    |     0.000000     |  0.175074   |3.400000e+03              | 5.000000 | 0.000000 |  0.000000   |0.000000    |        0.000000  |
|50%     | 0.000000     | 0.154181   | 52.000000     |          0.000000    |   0.366508    5.400000e+03     |  8.000000 | 0.000000  |  1.000000 | 0.000000 |           0.000000  |
|75%   | 0.000000   | 0.559046   |63.000000          |    0.000000  |     0.868254   |  8.249000e+03  |   11.000000  |0.000000   |   2.000000   | 0.000000    |        1.000000  |
|max    |1.000000    |50708.000000   |109.000000       |  98.000000  |329664.000000 |  3.008750e+06  |    58.000000 | 98.000000   |  54.000000|    98.000000   |        20.000000  |


#### 2. 空值

|  列名     |缺失值占比|
| ------ | ------ | 
|SeriousDlqin2yrs                        |0.000000
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

NumberOfDependents占比极其小，直接删除。
MonthlyIncome有接近20%的缺失，尝试用众数/中位数填充。

#### 3. 离群点

通过箱型图直观体现每一维数据的分布情况。可以看到图1、3、4、6~9中的箱被压缩的很严重，说明有部分数据十分远离中位数，分别对其进行考虑和处理。

![avatar](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/overview.png?raw=true)

a. 逾期数据

结合describe的结果，发现三个逾期数据（NumberOfTime30-59DaysPastDueNotWorse、NumberOfTime60-89DaysPastDueNotWorse、NumberOfTimes90DaysLate）具有非常类似的分布（在18~95之间都出现了巨大的gap，又有近300个样本出现在96~98之间），可以一起考量。分别采用删除/替换为18/替换为中位数三种方法，生成数据集1，2，3。

b. RevolvingUtilizationOfUnsecuredLines

此数据表示已贷金额和贷款额度的比值，远远大于1的数据不太正常。取1的十倍划线，删除异常值，生成数据集xx。


|  |Revolving<br>UtilizationOf<br>UnsecuredLines | SeriousDlqin2yrs
| ------ | ------ | -------|
|count              |             3321.000000   |    3321.000000
|mean               |              259.773362   |       0.372478
|std                |             1659.034074    |      0.483538
|min                 |               1.000059    |      0.000000
|25%                  |              1.019996    |      0.000000
|50%                  |              1.074633    |      0.000000
|75%                  |              1.301096    |      1.000000
|max                  |          50708.000000     |     1.000000

c. DebtRatio 和 MonthlyIncome

处理MonthlyIncome时发现，删除MonthlyIncome为空的数据前后，SeriousDlqin2yrs的均值发生了剧烈的变化（删除前是删除后的两倍）。可以认为

| |SeriousDlqin2yrs|	MonthlyIncome|
| ------ | ------ | -------|
|count	3750.000000	|185.000000|
|mean|	0.064267	|0.064865|
|std	|0.245260|	0.246956|
|min	|0.000000	|0.000000|
|25%	|0.000000	|0.000000|
|50%	|0.000000	|0.000000|
|75%	|0.000000	|0.000000|
|max	|1.000000	|1.000000|

将MonthlyIncome<=1或SeriousDlqin2yrs>=500的值全部替换为中位数各自的中位数，生成数据集XXXX

3. 引入模型评估

将上一步中生成的数据集分别使用RF、GBDT训练。结果如下：


可见，

- 评估处理方法
- 评估数据质量
- 评估importance

### 二、模型探索

1. Rondom Forest调参 
    - n_estimators
    - max_features
    - max_depth、min_sample_leaf、min_samples_split
    
2. GBDT调参

    GBDT的调参相对来说比较复杂，因为n_estimators和learning_rate需要一起调节。查阅资料learning_rate一般在0.1-0.3范围内，小于0.1亦可，单不要过大。于是选择0.01~0.35范围内，配合不同的n_estimators进行粗调。代码及折线图如下。
    
    
    # 粗调
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    n_estimators = np.linspace(10, 160, 12, endpoint=True)

![gbdt粗调](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/gbdt_rough_tuning_learning_rate.png?raw=true)

可见较好的auc出现在0.05附近。减小learning_rate的步进长度，在0.05周围继续网格搜索。同时略微调节n_estimators的范围。

有趣的是：

learning_rate越小，达到最佳AUC需要的n_estimators越大。也就是说，每次学习的残差越小，就需要叠加更多的树才能消除偏差。需要越好地体现了shrinkage的思想。

learning_rate越大，小步逼近正确值。learning_rate越大，需要的n_estimators就越小。但是随着learning_rate的增大，明显可以看到Test_AUC跟随Train_AUC的程度越低。也就是说越容易过拟合。

 
    # 细调
    learning_rate = [0.03, 0.05, 0.08, 0.1, 0.13, 0.15]
    n_estimators = np.linspace(30, 210, 10, endpoint=True)
    
    
    
   fv
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

    GBDT：shallow trees （ weak learners，high bias, low variance） +  gradient 降低bias。调参调的是variance

    RF：fully grown trees（low bias, high variance） + bagging 降低variance。调参调的是bias

2. 本文以外的思考

3. 关于正确率评判方式的思考

4. 阈值探索

