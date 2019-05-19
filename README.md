# trying-on-machine-learning

## 综述：
尝试探索kaggle网站上信用卡用户分类的competition([详情链接](https://www.kaggle.com/c/GiveMeSomeCredit/overview))，进行数据探索->数据清洗->模型训练->模型评估等一系列操作，内容主要包括：


### 1. 数据探索：
    
   1.1 空值的处理
   
   特征NumberOfDependents空值比例极小（2.62%），直接删除此列为空的样本。
   
   特征MonthlyIncome有19.82%的缺失，通过计算其IV（0.07），决定删除此特征。
   
   1.2 离群点的处理
   
   综合分析各维度的分布情况，找出可能的异常值，使用不同的处理方式（删除/替换）生成6种数据集。
   使用粗略调参的模型分别训练数据集，对比评估选出了3种较为合适的处理方式，综合后生成最优数据集，用于接下来的模型训练。

### 2. 模型训练和对比分析

2.1 调参

分别对RF和GBDT进行调参，寻找最佳模型，并进行对比。具体调参细节见文章第二部分，过程中的收获和思考如下：

RF = fully grown tree（低bias高variance）+ Bagging （降低variance)
GBDT = shallow tree（高bias低variance）+ Boosting （降低bias）

调参的最终目的是要找到bias和variance平衡的那个点，就是在提高train_auc的同时，尽量保证test_auc的跟随性，
最终将参数固定在极值点附近，也就是过拟合的临界点附近。

从大的方面来说，我在调参过程中发现了一些比较重要的思路

**- 先调学习器参数，再调集成参数**

调参的顺序非常重要。个人认为可以先调节学习器参数，因为集成参数可以参照经验，暂且设置得富裕一些（例如先将n_estimators设置得大一些，learning_rate设置得小一些），
这样会加大训练的时间，但不会过分影响模型的性能。将学习器参数调节得差不多之后，再去调节集成参数。

**- 先粗调再细调**

比如RF中的max_depth和min_samples_split/min_samples_leaf都是用来防止过拟合的。比较而言，max_depth的控制粒度比较粗，但是好调；后二者粒度细，但是很难把握。
要像尽量达到较好的正确率，应该尽量通过后两者去约束树停止生长的条件。但是直接调节后两者比较难。
于是我采取的方法是先找到max_depth的极值点，对模型有个大致的认知。再稍稍提高max_depth，用min_samples_split进行更细粒度的控制，将bias和variance的平衡点尽量往上推。

GBDT因为超参数的存在，网格搜索比较复杂，更应该遵循先粗调再细调的思路。



2.2 模型间的对比
    
   - 抗噪：人为引入噪声，对比二者的噪声能力。RF > GBDT
   - 速度：训练速度 RF > GBDT; 分类速度 GBDT>RF
   - 准确度：对competition提供的预测样本进行分类并上传，评估分类准确度。GBDT 整体略好
   - 调参: RF的调参更为简单
   
2.3 可以继续探索的问题。

因为时间有限，文章最后罗列了一些可以继续探究的问题，以及自己的猜测。

### 3. 结果：

将测试数据分测试集和验证集，交叉验证算出AUC-ROC的平均值，验证集大致在0.864左右，GBDT略好于RF。

使用对应模型对测试集进行分类，将分类结果提交，网站评分如下图（competition第一名为0.8695）

<img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/my_score.jpg?raw=true" width="75%" >

### 4. 文件列表

    comparator.py  ------------ 对比器，用于生成各模型/数据集的ROC列表、对比图
    data_explore.py  ---------- 数据处理
    rf_tuning.py  ------------- 随机森林调参 
    gbdt_tuning.py  ----------- GBDT调参  
    iv_calculator.py ---------- IV计算器
    gbdt_vs_rf ---------------- 调参后模型对比脚本
   
   
   <br/>
   
## 具体处理和数据支撑

### 一、数据探索

   共11维数据，150000个样本。通过describe函数查看大致情况，并加以分析。主要处理如下。

<!--#### 1. 数据描述

|           |  SeriousDlqin2yrs | Revolving<br>Utilization<br>OfUnsecuredLines  |age | NumberOfTime<br>30-59Days<br>PastDueNotWorse    |  DebtRatio | MonthlyIncome | NumberOfOpen<br>CreditLines<br>AndLoans|NumberOfTimes<br>90DaysLate  |NumberRealEstate<br>LoansOrLines|NumberOfTime<br>60-89Days<br>PastDueNotWorse | NumberOfDependents  |
| ------ | ------ | ------ | ----- | ---------| -----------|------|-------|------|--------|----|-----|
|count  |150000.000000  |150000.000000 |  150000.000000     |  150000.000000 | 150000.000000  |  1.202690e+05 |150000.000000  | 150000.000000 |   150000.000000 | 150000.000000  |     146076.000000 | 
|mean  |    0.066840 |    6.048438  |  52.295207     |    0.421033    | 353.005076   |   6.670221e+03        | 8.452760   |0.265973      |  1.018240  |  0.240387     |       0.757222  |
|std      |   0.249746   | 249.755371   |14.771866   |   4.192781   | 2037.818523  |    1.438467e+04     |  5.145951   |  4.169304   | 1.129771   | 4.155179     |       1.115086  |
|min  |  0.000000   |  0.000000   | 0.000000        |       0.000000      | 0.000000   |    0.000000e+00         |  0.000000  | 0.000000 |   0.000000 |    0.000000  |          0.000000  |
|25%     | 0.000000    | 0.029867   |  41.000000    |     0.000000     |  0.175074   |3.400000e+03              | 5.000000 | 0.000000 |  0.000000   |0.000000    |        0.000000  |
|50%     | 0.000000     | 0.154181   | 52.000000     |          0.000000    |   0.366508    |5.400000e+03     |  8.000000 | 0.000000  |  1.000000 | 0.000000 |           0.000000  |
|75%   | 0.000000   | 0.559046   |63.000000          |    0.000000  |     0.868254   |  8.249000e+03  |   11.000000  |0.000000   |   2.000000   | 0.000000    |        1.000000  |
|max    |1.000000    |50708.000000   |109.000000       |  98.000000  |329664.000000 |  3.008750e+06  |    58.000000 | 98.000000   |  54.000000|    98.000000   |        20.000000  |-->


#### 1. 空值

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

为了确定处理方法，选择合适的分组，编写代码计算IV，结果见下图。最终IV = 0.07039，说明MonthlyIncome与分类结果的关联性比较低，决定直接删除此维度的数据

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


#### 2. 离群点

通过describe函数和箱型图分析每一维数据的分布情况。可以看到图1、3、4、6~9中的箱被压缩的很严重，说明有部分数据十分远离中位数，分别对其进行考虑和处理。

![avatar](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/overview.png?raw=true)

**a. DebtRatio 和 MonthlyIncome**

处理MonthlyIncome时发现，删除MonthlyIncome为空的数据前后，DebtRatio的均值发生了剧烈的变化（删除前是删除后的两倍）。
猜测DebtRatio离群点和MonthlyIncome为空的样本存在大量重叠，对此类数据的真实度产生怀疑。选取95分位点打印信息如下：

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


**b. 逾期数据**

结合describe的结果，发现三个逾期数据（NumberOfTime30-59DaysPastDueNotWorse、NumberOfTime60-89DaysPastDueNotWorse、NumberOfTimes90DaysLate）
具有非常类似的分布（在18至95之间都出现了巨大的gap，又有近300个样本出现在96至98之间），可以一起考量。

分别采用删除/替换为18两种方法，生成数据集"overdue outliers replaced"和"overdue outliers removed"。

**c. RevolvingUtilizationOfUnsecuredLines**

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

<br/>

#### 3. 引入模型评估

将上一步中生成的数据集分别使用粗略调参的RF进行严重。结果如下：

     --- Sorted Results ---
    ('RF', 'debt ratio outliers removed') --> AUC: 0.8606 (+/- 0.0045)
    ('RF', 'debt ratio outliers replaced') --> AUC: 0.8601 (+/- 0.0052)
    ('RF', 'overdue outliers removed') --> AUC: 0.8598 (+/- 0.0074)
    ('RF', 'utilization outliers removed') --> AUC: 0.8592 (+/- 0.0036)
    ('RF', 'missing data processed') --> AUC: 0.8576 (+/- 0.0018)
    ('RF', 'overdue outliers replaced') --> AUC: 0.8576 (+/- 0.0050)

可见，"debt ratio outliers replaced"、"debt ratio outliers removed"、"overdue outliers removed"、
"utilization outliers removed"这四个数据集的表现优于仅处理空值的"missing data processed"。

考虑采用其对应的处理方式生成最佳训练集，并使用此数据集进行接下来的调参探索。最佳训练集的训练结果如下，确实优于其他所有数据集，可以佐证此处理方式的合理性：

    ('RF', 'best_data') --> AUC: 0.8647 (+/- 0.0041)


### 二、模型探索

主要涉及参数如下：

| | RF参数 | GBDT参数 | 
|-------| ------ | ------ | 
| 集成关参数|n_estimators<br>max_features|n_estimators和learning_rate<br>subsample<br>loss|
| 学习器参数|max_depth<br>min_sample_leaf<br>min_samples_split|同RF|
|其他参数|criterion|subsample<br>loss|

#### 1. RF调参 

RF调参比较简单，因为参数之间的相互影响比较小，可以直接对单一参数进行网格搜索。主要有以下三个层面的参数需要调节：

   - n_estimators：对训练时间的影响最大，与时间基本呈线性关系。
   - max_features：'auto', 'sqrt', 'log'差距极小。 猜测是因为本数据集维度比较低（10），所以直接使用'auto'即可。
   - max_depth、min_sample_leaf、min_samples_split：体现了单棵树停止生长的条件，三者的作用都是防止过拟合。
    其中max_depth效果最显著，调起来最方便。如果在max_depth选择了最佳值之后，仍然需要提高正确率，可以略略放大max_depth，再对min_sample_leaf、min_samples_split用于精细调节。
    
   调参顺序：
    
   max_depth -> n_estimators -> min_sample_leaf或min_samples_split
    
   **- max_depth**
   
   对max_depth进行网格搜索，见下图。max_depth达到8的时候，AUC基本达到最大值。在8-30之间测试集AUC还在上升，而验证集已经不再上升，
   显然此时存在过拟合。
    
   <img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/rf_tuning_depth(split=500).png?raw=true" width="50%">
   
   <br />
    
   **- n_estimators**
   
   n_estimators的网格搜索结果见下图。n_estimators达到64的时候，AUC基本达到最大值。同时注意到n_estimate与训练耗时基本呈正比。

   <img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/rf_tuning_n_estimate.png?raw=true" width="80%" height="30%">
   
   <br />
    
    
   **- min_samples_leaf**
    
   放大max_depth至14，对min_samples_leaf进行网格搜索。可以看到极值点出现在100附近，即min_samples_leaf<100时出现了过拟合。此极值点处的AUC大于max_depth调节之后的AUC，
   可见此操作成功延迟了过拟合的出现，提高了AUC。
   
   **注意：一般min_samples_leaf的取值应该在0.5%-1%之间，但是此数据集存在非常严重的unbalanced现象，所以min_samples_leaf偏小。**
       
   <img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/rf_tuning_leaf(depth=16).png?raw=true" width="50%" >

   最终选择 n_estimators=100, max_depth=16, max_features='auto', min_samples_leaf=100 的组合作为RF的最佳参数。

#### 2. GBDT调参
    
   **调参顺序：**

   max_depth —> min_samples_split/min_samples_leaf -> n_estimators + learning_rate -> subsample -> loss

   即基学习器参数 -> Boosting相关参数 -> 其他参数
    
   前两步的调节的方法与RF基本一致。主要是注意相比RF，**max_depth要小，min_samples_split/min_samples_leaf要大一些**。
   原因是GBDT并不要求每一棵树的预估结果都很准确，反正可以通过不断减少残差去接近正确结果，提高每一棵树的抗噪能力更加重要。
   
   
   **- n_estimators + learning_rate**
   
   调参的主要难度在n_estimators和learning_rate这一步，因为这两个参数需要一起调节。
   
   查阅资料learning_rate一般在0.1-0.3范围内，小于0.1亦可，但不要过大。于是选择0.01~0.35范围内，配合不同的n_estimators进行粗调。代码及折线图如下。
    
    #粗调
    learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    n_estimators = np.linspace(10, 160, 12, endpoint=True)
    

   ![gbdt粗调](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/gbdt_rough_tuning_learning_rate.png?raw=true)
   
   这两者之间的关系很好地体现了残差学习的思想：learning_rate越小，达到最佳AUC需要的n_estimators越大（意味着训练、分类的时间越大）。也就是说，每次学习的残差越小，就需要叠加更多的树才能消除偏差；
   learning_rate越大，需要的n_estimators就越小。但是随着learning_rate的增大，明显可以看到test_auc和train_auc之间的夹角越大，也就是测试集正确率跟随训练集的能力越低。此时出现了过拟合。

   上图可见较好的auc出现在0.05附近。减小learning_rate的步进长度，在0.03-0.1之间继续网格搜索，结果如下图。
   可以看到当learning_rate>0.05后，test_auc的跟随性开始变差。最终选择learning_rate=0.05，n_estimators=200。

    # 细调
    learning_rate = [0.03, 0.05, 0.08, 0.1, 0.13, 0.15]
    n_estimators = np.linspace(30, 210, 10, endpoint=True)
    
   ![gbdt细调](https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/gbdt_delicate_tuning_learning_rate.png?raw=true)



   **- subsample**
   
   在(0.5, 1)之间步进搜索，AUC折线图如下。整体而言subsamples的取值对结果的影响不大，test_auc对train_auc一直有比较好的跟随性。
   猜测是因为离群点滤得比较干净，之前的防过拟合参数也比较合适。最终选取 subsample=0.85。
   
   <img src="https://github.com/IggyGao/trying-on-machine-learning/blob/master/pictures/gbdt_tuning_subsample.png?raw=true" width="60%" >
  
    
#### 3. 模型对比

   **a. 制造噪声，对比抗噪能力**
    
   参考模型计算出的importance，选择在RevolvingUtilizationOfUnsecuredLines这个比较重要的维度上引入噪声。
    随机抽取5%的样本，修改这个维度的值。
    
   首先比较调参前后的GBDT。相比调参前，调参后AUC-ROC下降程度很小，可见上文的调参工作确实起到了抗噪声的作用。
    
    
     --- Sorted Results ---
    ('tuned GBDT', 'data') --> AUC: 0.8628 (+/- 0.0045)
    ('tuned GBDT', 'outliers added') --> AUC: 0.8627 (+/- 0.0029)
    ('default GBDT', 'data') --> AUC: 0.8623 (+/- 0.0025)
    ('default GBDT', 'outliers added') --> AUC: 0.8576 (+/- 0.0026)
    
    
   然后比较调参后的RF和GBDT。RF的AUC-ROC下降程度小于GBDT，可见RF的抗噪声能力更强。
    
    
     --- Sorted Results ---
    ('tuned GBDT', 'data') --> AUC: 0.8640 (+/- 0.0020)
    ('tuned RF', 'data') --> AUC: 0.8634 (+/- 0.0009)
    ('tuned RF', 'outliers data') --> AUC: 0.8627 (+/- 0.0045)
    ('tuned GBDT', 'outliers added') --> AUC: 0.8611 (+/- 0.0040)
    
   **b. 对比训练/分类速度**
   
     --- Time Spent ---
    RF train costs -- 14.72s 
    RF test costs -- 1.01s 
    GBDT train costs -- 90.18s 
    GBDT test costs -- 0.60s 
    
   可见GBDT的训练耗时远远大于RF，但是分类耗时相差不多。
   
   GBDT的训练耗时大，是因为GBDT中树的训练是串行的，并且一般会采用较小的learnning_rate防止跃过最优解，所以叠加的树的规模也会大幅增加。
   XGBoost在feature层面上采用了预排序，将训练速度提高了很多。

   **c. 调参对比**
   
   对比上文的调参工作，因为hyperparameters的存在，GBDT的调参难度远远大于RF。不过多处资料表明，如果调参得当，GBDT的分类结果会好于RF。
    
    
#### 4. 可以继续探索的问题
   
   - min_samples_split和min_samples_leaf
   
   之前已经描述并且证明了max_depth与这二者的关系。但是如果需要跟精细的调参，这二者之间又应该如何调节？个人猜测，
   在N、P样本比较均衡的情况下，调节这两者中的任何一个都可以（min_samples_leaf = n 应该等价于 min_samples_split = 2n的情况）。
   但是在非常不均衡的样本下，可能需要对二者进行更加精细的选择，比如缩小min_samples_split的值，以适应较小的P样本占比。
   实际情况是不是符合猜测，合适的调节又会对分类正确率有多大的提高，还需要更多实验来验证。
   
   - 模型的评价方式
   
   competition中默认的评分方式是AUC-ROC，但是实际上信用卡评分可能是一个比较unbalanced的数据集。如果实际情况中N样本的占比远大于训练集，
   使用AUC-ROC的评估可能会过于乐观。而且考虑到模型评分可能是一个初筛手段，应该更关心P样本的预测正确率。
   所以如果有时间的话，可以考虑使用AUC-PR来尝试评估模型，也许会更为合适。
   
   



