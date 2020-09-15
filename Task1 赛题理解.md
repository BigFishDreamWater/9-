## Task1 赛题理解

/***以下部分直接复制学习资料以充字数，个人理解在最下方***/

Tip:本次新人赛是Datawhale与天池联合发起的0基础入门系列赛事第四场 —— 零基础入门金融风控之贷款违约预测挑战赛。 赛题以金融风控中的个人信贷为背景，要求选手根据贷款申请人的数据信息预测其是否有违约的可能，以此判断是否通过此项贷款，这是一个典型的分类问题。通过这道赛题来引导大家了解金融风控中的一些业务背景，解决实际问题，帮助竞赛新人进行自我练习、自我提高。

项目地址：https://github.com/datawhalechina/team-learning-data-mining/tree/master/FinancialRiskControl

比赛地址：https://tianchi.aliyun.com/competition/entrance/531830/introduction

### 1.1 学习目标

理解赛题数据和目标，清楚评分体系。

完成相应报名，下载数据和结果提交打卡（可提交示例结果），熟悉比赛流程

### 1.2 了解赛题

- 赛题概况
- 数据概况
- 预测指标
- 分析赛题

### 1.2.1 赛题概况

##### 比赛要求参赛选手根据给定的数据集，建立模型，预测金融风险。

赛题以预测金融风险为任务，数据集报名后可见并可下载，该数据来自某信贷平台的贷款记录，总数据量超过120w，包含47列变量信息，其中15列为匿名变量。为了保证比赛的公平性，将会从中抽取80万条作为训练集，20万条作为测试集A，20万条作为测试集B，同时会对employmentTitle、purpose、postCode和title等信息进行脱敏。

### 1.2.2 数据概况

一般而言，对于数据在比赛界面都有对应的数据概况介绍（匿名特征除外），说明列的性质特征。了解列的性质会有助于我们对于数据的理解和后续分析。 Tip:匿名特征，就是未告知数据列所属的性质的特征列。

train.csv

- id 为贷款清单分配的唯一信用证标识
- loanAmnt 贷款金额
- term 贷款期限（year）
- interestRate 贷款利率
- installment 分期付款金额
- grade 贷款等级
- subGrade 贷款等级之子级
- employmentTitle 就业职称
- employmentLength 就业年限（年）
- homeOwnership 借款人在登记时提供的房屋所有权状况
- annualIncome 年收入
- verificationStatus 验证状态
- issueDate 贷款发放的月份
- purpose 借款人在贷款申请时的贷款用途类别
- postCode 借款人在贷款申请中提供的邮政编码的前3位数字
- regionCode 地区编码
- dti 债务收入比
- delinquency_2years 借款人过去2年信用档案中逾期30天以上的违约事件数
- ficoRangeLow 借款人在贷款发放时的fico所属的下限范围
- ficoRangeHigh 借款人在贷款发放时的fico所属的上限范围
- openAcc 借款人信用档案中未结信用额度的数量
- pubRec 贬损公共记录的数量
- pubRecBankruptcies 公开记录清除的数量
- revolBal 信贷周转余额合计
- revolUtil 循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额
- totalAcc 借款人信用档案中当前的信用额度总数
- initialListStatus 贷款的初始列表状态
- applicationType 表明贷款是个人申请还是与两个共同借款人的联合申请
- earliesCreditLine 借款人最早报告的信用额度开立的月份
- title 借款人提供的贷款名称
- policyCode 公开可用的策略_代码=1新产品不公开可用的策略_代码=2
- n系列匿名特征 匿名特征n0-n14，为一些贷款人行为计数特征的处理

### 1.2.3 预测指标

竞赛采用AUC作为评价指标。AUC（Area Under Curve）被定义为 ROC曲线 下与坐标轴围成的面积。

##### 分类算法常见的评估指标如下：

1、混淆矩阵（Confuse Matrix）

- （1）若一个实例是正类，并且被预测为正类，即为真正类TP(True Positive )
- （2）若一个实例是正类，但是被预测为负类，即为假负类FN(False Negative )
- （3）若一个实例是负类，但是被预测为正类，即为假正类FP(False Positive )
- （4）若一个实例是负类，并且被预测为负类，即为真负类TN(True Negative )

2、准确率（Accuracy） 准确率是常用的一个评价指标，但是不适合样本不均衡的情况。 $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

3、精确率（Precision） 又称查准率，正确预测为正样本（TP）占预测为正样本(TP+FP)的百分比。 $$Precision = \frac{TP}{TP + FP}$$

4、召回率（Recall） 又称为查全率，正确预测为正样本（TP）占正样本(TP+FN)的百分比。 $$Recall = \frac{TP}{TP + FN}$$

5、F1 Score 精确率和召回率是相互影响的，精确率升高则召回率下降，召回率升高则精确率下降，如果需要兼顾二者，就需要精确率、召回率的结合F1 Score。 $$F1-Score = \frac{2}{\frac{1}{Precision} + \frac{1}{Recall}}$$

6、P-R曲线（Precision-Recall Curve） P-R曲线是描述精确率和召回率变化的曲线

[![p-r](https://camo.githubusercontent.com/25688baabbff569136e19f04c812ed80af778351/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363132352e706e67)](https://camo.githubusercontent.com/25688baabbff569136e19f04c812ed80af778351/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363132352e706e67)

7、ROC（Receiver Operating Characteristic）

- ROC空间将假正例率（FPR）定义为 X 轴，真正例率（TPR）定义为 Y 轴。

TPR：在所有实际为正例的样本中，被正确地判断为正例之比率。 $$TPR = \frac{TP}{TP + FN}$$ FPR：在所有实际为负例的样本中，被错误地判断为正例之比率。 $$FPR = \frac{FP}{FP + TN}$$

[![roc.png](https://camo.githubusercontent.com/2dfea351fa5eac42caab9e716aa76a20553e4103/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363132342e706e67)](https://camo.githubusercontent.com/2dfea351fa5eac42caab9e716aa76a20553e4103/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363132342e706e67)

8、AUC(Area Under Curve) AUC（Area Under Curve）被定义为 ROC曲线 下与坐标轴围成的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围在0.5和1之间。AUC越接近1.0，检测方法真实性越高;等于0.5时，则真实性最低，无应用价值。

##### 对于金融风控预测类常见的评估指标如下:

1、KS(Kolmogorov-Smirnov) KS统计量由两位苏联数学家A.N. Kolmogorov和N.V. Smirnov提出。在风控中，KS常用于评估模型区分度。区分度越大，说明模型的风险排序能力（ranking ability）越强。 K-S曲线与ROC曲线类似，不同在于

- ROC曲线将真正例率和假正例率作为横纵轴
- K-S曲线将真正例率和假正例率都作为纵轴，横轴则由选定的阈值来充当。 公式如下： $$KS=max(TPR-FPR)$$ KS不同代表的不同情况，一般情况KS值越大，模型的区分能力越强，但是也不是越大模型效果就越好，如果KS过大，模型可能存在异常，所以当KS值过高可能需要检查模型是否过拟合。以下为KS值对应的模型情况，但此对应不是唯一的，只代表大致趋势。

| KS（%） | 好坏区分能力         |
| ------- | -------------------- |
| 20以下  | 不建议采用           |
| 20-40   | 较好                 |
| 41-50   | 良好                 |
| 51-60   | 很强                 |
| 61-75   | 非常强               |
| 75以上  | 过于高，疑似存在问题 |

2、ROC

3、AUC

### 1.2.4. 赛题流程

[![1_1.png](https://camo.githubusercontent.com/baf3abb3b945608363c4e7a0be0a8a158629463b/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363131302e706e67)](https://camo.githubusercontent.com/baf3abb3b945608363c4e7a0be0a8a158629463b/68747470733a2f2f696d672d626c6f672e6373646e696d672e636e2f32303230303931333031303232363131302e706e67)





1.导入数据：

![image-20200915194705488](C:\Users\牧羊人\AppData\Roaming\Typora\typora-user-images\image-20200915194705488.png)

先是对着题目中的字段含义和实际数据做一个对比理解。

主要难以理解的字段有如下：

1. verificationStatus 验证状态

2. ficoRangeLow借款人在贷款发放时的fico所属的下限范围

3. ficoRangeHigh 借款人在贷款发放时的fico所属的上限范围

4. pubRecBankruptcies 公开记录清除的数量

   

![image-20200915195636913](C:\Users\牧羊人\AppData\Roaming\Typora\typora-user-images\image-20200915195636913.png)

verificationStatus，百度了下也没有明白啥意思，就看了下分布，只有3个分类，且分布相对均匀，影响应该不大。

ficoRangeLow  主要就是fico分数，是美国费埃哲公司，也是最大的个人信用评分机构作出对个人信用作出的一个评分，具有一定参考价值。

范围位于300- 850分之间。分数越高, 说明客户的信用风险越小。但是分数本身并不能说明一个客户是好还是坏,贷款方通常会将分数作为参考, 来进行贷款决策。每个贷款方都会有自己的贷款策略和标准, 并且每种产品都会有自己的风险水平, 从而决定了可以接受的信用分数水平。一般地说, 如果借款人的信用评分达到680 分以上, 贷款方就可以认为借款人的信用卓著,可以毫不迟疑地同意发放贷款。如果借款人的信用评分低于620 分, 贷款方或者要求借款人增加担保, 或者干脆寻找各种理由拒绝贷款。

但是看这个数据集的分布：

![image-20200915200147183](C:\Users\牧羊人\AppData\Roaming\Typora\typora-user-images\image-20200915200147183.png)

均值和最低值都远远高于620分，美国人的评分真的是仅供参考了。

pubRecBankruptcies 公开记录清除的数量：这个貌似在公开渠道能查到的申请破产次数...

![image-20200915200438796](C:\Users\牧羊人\AppData\Roaming\Typora\typora-user-images\image-20200915200438796.png)

像这种破产十二次的多半就是风险很高了。

好奇查看下这个是不是违约了。

![image-20200915203939251](C:\Users\牧羊人\AppData\Roaming\Typora\typora-user-images\image-20200915203939251.png)

好吧，是我误判了。

人家并没有违约。



初步预估重要变量主要在annualincome 年收入，delinquency_2years 去年2年违约比数，openAcc 借款人信用档案中未结信用额度的数量



接着网上搜索了下类似比赛的TOP分享，搜索到了kaggle上16年的一个比赛TOP1的分享。

链接如下。

https://nycdatascience.com/blog/student-works/kaggle-predict-consumer-credit-default/



最后的分数在0.86.....

可以非常的高了，相比现在排行榜0.75来说。

等2天学习完再来总结。

