# trying-on-machine-learning

##综述：
尝试探索kaggle网站上信用卡用户分类的competition([地址超链接](https://www.kaggle.com/c/GiveMeSomeCredit/overview))，进行数据探索-数据清洗-模型训练-模型评估等一系列操作。内容主要包括：


#####1. 数据探索：
- 处理空值。分析空值占比，占比极少的直接删除，较多的进行填充。
- 处理离群点。通过箱型图+knn找出可能的异常值。

对第二步中找出的异常值分别处理，处理后的数据为数据集1-5。使用模型分别训练数据集1-5，对比评估，从而选出合适的数据处理方式，用于接下来的模型训练。

#####2. 模型训练：
分别对随机森林和GBDT进行调参，考虑ROC-AOC、速度、是否过拟合等方面分别找出较好的参数。并从预测准确性、抗噪声能力等方面对两种模型进行对比。

结果：
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
2. 离群点
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

