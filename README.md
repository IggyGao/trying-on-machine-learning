# trying-on-machine-learning

## 综述：
尝试探索kaggle网站上信用卡用户分类的competition([详情链接](https://www.kaggle.com/c/GiveMeSomeCredit/overview))，进行数据探索-数据清洗-模型训练-模型评估等一系列操作，内容主要包括：


#### 1. 数据探索：
    
   - 空值的处理
   
   NumberOfDependents空值比例极小（2.62%），直接删除此列为空的样本。
   
   MonthlyIncome有19.82%的缺失，通过计算其IV（0.07），决定删除此维度。
   
   - 离群点的处理
   
   综合分析各维度的分布情况，找出可能的异常值，使用不同的处理方式（删除/替换）生成6种数据集。
   使用粗略调参的模型分别训练数据集，对比评估选出了3种较为合适的处理方式，综合后生成最优数据集，用于接下来的模型训练。

#### 2. 模型训练和对比分析

a. 调参

对比不同参数下随机森林和GBDT模型的好坏，包括：分类的正确性（AUC_ROC）、速度、是否过拟合等，寻找到此数据集下较为合适的参数。涉及参数如下：

| | RF参数 | GBDT参数 | 
|-------| ------ | ------ | 
| Bagging/Boosting相关参数|n_estimators<br>max_features|n_estimators和learning_rate<br>subsample<br>loss|
| 学习器相关参数|max_depth<br>min_sample_leaf<br>min_samples_split|同RF|

b. 思考与结论

RF核心思想：fully grown trees（低bias高variance） + Bagging （降低variance）。

GBDT核心思想：shallow trees（高bias低variance） + Boosting （降低bias）


所以如果分类偏差较大，RF应该调节学习器相关参数，GBDT应该调节Boosting相关参数；

如果抗噪能力比较弱，RF应该调节Bagging参数，GBDT应该调节学习器相关参数。

b. 模型间的对比 
    
   - 抗噪能力：RF > GBDT
   - 速度: 训练速度RF > GBDT; 分类速度DBDT>RF
   - 分类准确度: GBDT 整体略好
   - 调参: RF更简单

#### 3. 结果：
将测试数据分测试集和验证集，交叉验证算出AUC-ROC的平均值，验证集大致在0.864左右，GBDT略好于RF。

使用对应模型对测试集进行分类，生成submission.csv提交，网站评分如下图（最高分为0.8695）

<img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/my_score.png?raw=true" width="70%" >

#### 4. 延伸思考

关于模型评估方式（AUC-ROC）的一些疑问和思考，见文章最后

#### 5. 文件列表

    comparator.py  ------------ 对比器，用于生成各模型/数据集的ROC列表和调参对比图
    data_explore.py  ---------- 数据清洗
    rf_tuning.py  ------------- 随机森林调参 
    gbdt_tuning.py  ----------- GBDT调参  
    iv_calculator.py ---------- IV计算器
   
## 具体处理和数据支撑

### 一、数据探索

#### 1. 数据描述

   共11维数据，describe看一下大致情况。加以分析。

|           |  SeriousDlqin2yrs | Revolving<br>Utilization<br>OfUnsecuredLines  |age | NumberOfTime<br>30-59Days<br>PastDueNotWorse    |  DebtRatio | MonthlyIncome | NumberOfOpen<br>CreditLines<br>AndLoans|NumberOfTimes<br>90DaysLate  |NumberRealEstate<br>LoansOrLines|NumberOfTime<br>60-89Days<br>PastDueNotWorse | NumberOfDependents  |
| ------ | ------ | ------ | ----- | ---------| -----------|------|-------|------|--------|----|-----|
|count  |150000.000000  |150000.000000 |  150000.000000     |  150000.000000 | 150000.000000  |  1.202690e+05 |150000.000000  | 150000.000000 |   150000.000000 | 150000.000000  |     146076.000000 | 
|mean  |    0.066840 |    6.048438  |  52.295207     |    0.421033    | 353.005076   |   6.670221e+03        | 8.452760   |0.265973      |  1.018240  |  0.240387     |       0.757222  |
|std      |   0.249746   | 249.755371   |14.771866   |   4.192781   | 2037.818523  |    1.438467e+04     |  5.145951   |  4.169304   | 1.129771   | 4.155179     |       1.115086  |
|min  |  0.000000   |  0.000000   | 0.000000        |       0.000000      | 0.000000   |    0.000000e+00         |  0.000000  | 0.000000 |   0.000000 |    0.000000  |          0.000000  |
|25%     | 0.000000    | 0.029867   |  41.000000    |     0.000000     |  0.175074   |3.400000e+03              | 5.000000 | 0.000000 |  0.000000   |0.000000    |        0.000000  |
|50%     | 0.000000     | 0.154181   | 52.000000     |          0.000000    |   0.366508    |5.400000e+03     |  8.000000 | 0.000000  |  1.000000 | 0.000000 |           0.000000  |
|75%   | 0.000000   | 0.559046   |63.000000          |    0.000000  |     0.868254   |  8.249000e+03  |   11.000000  |0.000000   |   2.000000   | 0.000000    |        1.000000  |
|max    |1.000000    |50708.000000   |109.000000       |  98.000000  |329664.000000 |  3.008750e+06  |    58.000000 | 98.000000   |  54.000000|    98.000000   |        20.000000  |


#### 2. 空值

|  列名     |缺失值占比（%）|
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

MonthlyIncome有接近20%的缺失，缺失量很大，可以考虑用中位数/平均值填充 或 直接删除此列数据。

为了确定处理方法，选择合适的分组，编写代码计算IV，结果见下图。最终IV = 0.07，说明MonthlyIncome与分类结果的关联性比较低，决定直接删除此维度的数据

 |Value  |     sample rate   | Distribution Good | Distribution Bad |      Sub     |  WoE   |     IV|
 | ------ | ------|------ | ------|-------|----------|-------|
|      0-2000    |0.09  |     0.071249         | 0.091163 |-0.019914 |-0.246469  |0.004908
|   2000-3000     |   0.10|  0.080343         | 0.118991 |-0.038647 |-0.392734  |0.015178
|   3000-4000      |   0.12 | 0.097232         | 0.135248 |-0.038016 |-0.330009 |0.012546
|   4000-5000    |     0.12|0.097325         | 0.118392 |-0.021067 |-0.195945 |0.004128
|   5000-6000      |     0.11|0.293962         | 0.261520 |0.032442  |0.116939  |0.003794
|   6000-7000      |    0.09|0.076471          | 0.072511 |0.003960  |0.053171 |0.000211
|   7000-9000      |     0.13|0.112285         | 0.089567 |0.022718 |0.226053  |0.005135
|   9000-12000      |     0.11|0.094117         | 0.061839 |0.032278  |0.420006  |0.013557
|  12000以上      |    0.09| 0.077014         | 0.050768 |0.026246  |0.416725  |0.010937

IV =  0.07039407793853665

#### 3. 离群点

通过箱型图直观体现每一维数据的分布情况。可以看到图1、3、4、6~9中的箱被压缩的很严重，说明有部分数据十分远离中位数，分别对其进行考虑和处理。

![avatar](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/overview.png?raw=true)

a. 逾期数据

结合describe的结果，发现三个逾期数据（NumberOfTime30-59DaysPastDueNotWorse、NumberOfTime60-89DaysPastDueNotWorse、NumberOfTimes90DaysLate）
具有非常类似的分布（在18~95之间都出现了巨大的gap，又有近300个样本出现在96~98之间），可以一起考量。分别采用删除/替换为18两种方法，生成数据集"overdue outliers replaced"和"overdue outliers removed"。

b. RevolvingUtilizationOfUnsecuredLines

此数据表示已贷金额和贷款额度的比值，远远大于1的数据不太正常。取1的十倍划线，删除异常值，生成数据集"utilization outliers removed"。

<!--|  |Revolving<br>UtilizationOf<br>UnsecuredLines | SeriousDlqin2yrs
| ------ | ------ | -------|
|count              |             3321.000000   |    3321.000000
|mean               |              259.773362   |       0.372478
|std                |             1659.034074    |      0.483538
|min                 |               1.000059    |      0.000000
|25%                  |              1.019996    |      0.000000
|50%                  |              1.074633    |      0.000000
|75%                  |              1.301096    |      1.000000
|max                  |          50708.000000     |     1.000000-->

c. DebtRatio 和 MonthlyIncome

处理MonthlyIncome时发现，删除MonthlyIncome为空的数据前后，SeriousDlqin2yrs的均值发生了剧烈的变化（删除前是删除后的两倍）。
可见DebtRatio离群点和MonthlyIncome为空的样本存在大量重叠，对此类数据的真实度产生怀疑。选取95分位点打印信息如下：

 |          |DebtRatio | MonthlyIncome
| ------ | ------ | -------|
|count    |7836.000000     |399.000000
|mean     |4330.529862       |0.087719
|std      |7712.385814       |0.283241
|min      |2382.000000       |0.000000
|25%      |2824.000000       |0.000000
|50%      |3424.500000       |0.000000
|75%      |4535.000000       |0.000000
|max    |329664.000000       |1.000000


将DebtRatio>=2382的值全部替换为2382，生成数据集"debt ratio outliers replaced"

将DebtRatio>=2382的样本删除，生成数据集"debt ratio outliers removed"

3. 引入模型评估

将上一步中生成的数据集分别使用粗略调参的RF进行严重。结果如下：

     --- Sorted Results ---
    ('RF', 'debt ratio outliers removed') --> AUC: 0.8606 (+/- 0.0045)
    ('RF', 'debt ratio outliers replaced') --> AUC: 0.8601 (+/- 0.0052)
    ('RF', 'overdue outliers removed') --> AUC: 0.8598 (+/- 0.0074)
    ('RF', 'utilization outliers removed') --> AUC: 0.8592 (+/- 0.0036)
    ('RF', 'missing data processed') --> AUC: 0.8576 (+/- 0.0018)
    ('RF', 'overdue outliers replaced') --> AUC: 0.8576 (+/- 0.0050)

可见，"debt ratio outliers replaced"、"debt ratio outliers removed"、"overdue outliers removed"、
"utilization outliers removed"这四个数据集的表现优于仅仅处理空值的"missing data processed"。
考虑采用其对应的处理方式生成最佳训练集，并使用此数据集进行接下来的调参探索。最佳训练集的训练结果如下，确实优于其他所有数据集，可以佐证此处理方式的合理性：

    ('RF', 'best_data') --> AUC: 0.8647 (+/- 0.0041)


### 二、模型探索

1. RF调参 

    RF调参比较简单，因为参数之间的相互影响比较小，可以直接对单一参数进行网格搜索。主要有以下三个层面的参数需要调节：

    - n_estimators：对训练时间的影响最大，与时间基本呈线性关系。
    
    - max_features：'auto', 'sqrt', 'log'差距极小。 猜测是因为本数据集维度比较低（10），所以直接使用'auto'即可。
    
    - max_depth、min_sample_leaf、min_samples_split：体现了单棵树停止生长的条件，三者的作用都是防止过拟合。
    其中max_depth效果最显著，调起来最方便。如果在max_depth选择了最佳值之后，仍然需要提高正确率，可以略略放大max_depth，再对min_sample_leaf、min_samples_split用于精细调节。
    
    调参顺序：
    
    max_depth -> n_estimators -> min_sample_leaf或min_samples_split
    
    首先调节max_depth，见下图。max_depth达到8的时候，AUC基本达到最大值。在8-30之间测试集AUC还在上升，而验证集已经不再上升，
    显然此时存在过拟合。
    
    ![rf_max_depth](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/rf_tuning_depth(split=500).png?raw=true)
    
    接着调节n_estimators，见下图。n_estimators达到64的时候，AUC基本达到最大值。n_estimate与训练耗时基本呈正比。

    ![rf_n_estimate](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/rf_tuning_n_estimate.png?raw=true)
    
    最后放大max_depth至14，对min_samples_leaf进行网格搜索。可以看到在曲线的前半段(<100)时，曲线是上升的，即过拟合得到了一定的抑制。
    
    ![rf_min_samples_leaf](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/rf_tuning_leaf(depth=16).png?raw=true)

    
2. GBDT调参

    GBDT调参的思路是基学习器参数 -> Boosting参数 -> 其他参数。基学习器参数。
    
    具体到参数上即为max_depth + min_samples_split -> _estimators + learning_rate -> subsample -> loss


   - n_estimators和learning_rate
    
   GBDT的调参相对来说比较复杂，因为n_estimators和learning_rate需要一起调节。查阅资料learning_rate一般在0.1-0.3范围内，小于0.1亦可，单不要过大。于是选择0.01~0.35范围内，配合不同的n_estimators进行粗调。代码及折线图如下。
    
    
    #粗调
    
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    n_estimators = np.linspace(10, 160, 12, endpoint=True)

![gbdt粗调](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/gbdt_rough_tuning_learning_rate.png?raw=true)

有趣的是这两者之间的关系很好地体现了残差学习的思想。

learning_rate越小，达到最佳AUC需要的n_estimators越大（意味着训练、分类的时间越大）。也就是说，每次学习的残差越小，就需要叠加更多的树才能消除偏差。

learning_rate越大，需要的n_estimators就越小。但是随着learning_rate的增大，明显可以看到Test_AUC和Train_AUC之间的夹角越大，也就是测试集正确率跟随训练集的能力越低。也就是说此时出现了过拟合。


上图可见较好的auc出现在0.05附近。减小learning_rate的步进长度，在0.03-0.1之间继续网格搜索，结果如下图。


    # 细调
    learning_rate = [0.03, 0.05, 0.08, 0.1, 0.13, 0.15]
    n_estimators = np.linspace(30, 210, 10, endpoint=True)
    
![gbdt细调](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/gbdt_delicate_tuning_learning_rate.png?raw=true)

当learning_rate<0.05时，Test_ROC曲线一直保持着比较好的跟随性。当learning_rate>0.05后，test_auc的跟随性开始变差。本来打算选择0.05以下的learning_rate，但是我考虑到此时的subsample = 1，还没有发挥其抗过拟合的作用，所以我尝试选择了0.05作为learning_rate进行接下来的探索。看看能不能通过调节其他参数，让这条曲线上扬，从而提高auc，降低n_estimators（也就是降低时间）。

   - subsample
   
   采用learning_rate=0.05，二重循环同时步进n_estimators和subsample 
   
    n_estimators = [30, 90, 120, 150, 180, 210, 240, 270, 290, 320]
    subsample = np.linspace(0.5, 1, 6, endpoint=True)
    
    
   结果如图。
   
   - loss
   
   - max_depth
   
   max_depth越小，越能降低过拟合。
  
    
3. 模型对比

    a. 制造噪声，对比抗噪能力
    
    根据上文中提到的importance，选择在xxx和xxx这两个比较重要的维度上引入噪声。
    分别随机抽取4%的样本，修改这两个维度的值。
    然后用默认参数的GBDT和RF（分别称为default GBDT和default RF），上文中调好参数的GBDT和RF（分别称为tuned GBDT和tuned RF）进行训练，结果如下：
   
   可以看到
    
    -  无论是否经过GBDT的AUC-ROC下降程度大于DF。可见GBDT的抗噪声能力更强。
    - 调参之后，相同模型AUC-ROC下降程度小于未调参。可见之前的调参工作确实起到了抗噪声的作用。
    
    b. 对比训练/分类速度（对比同级别？？？还是对比调参之后的？？）
    RF训练速度更快，但是分类速度慢

    c. 调参对比
    对比上文的调参工作，因为hyperparams的存在，GBDT的调参难度远远大于RF。不过多处资料表明，如果调参得当，GBDT的性能会好于RF。本文也是如此。


## 其他一些思考

2. 本文以外的思考
    
    competition中默认的评分方式是ROC。



