# SameQuestion
基于Adversarial Attack的问题等价性判别比赛

## 使用BERT+Large中文预训练模型
思路很简单，直接使用BERT模型跑分类；

更详细的开发日志,看这里[dev_readme.md](./dev_readme.md)

### 使用K Fold进行模型融合 当前最好成绩 0.90840

完成 K-Fold [2019/11/22]

选择K=5, 训练5个数据模型，对结果进行加权平均，提交后

```
48	↑43	xmxoxo 0.90840	7
```

### 目前最好成绩 0.8992 [2019/11/20]

使用large模型

数据：10.1W , num_train_epochs=3.0
生成提交数据，提交得到：0.8992


### Todo: 数据增强 [2019/11/15]

修改了preProcess.py 添加了"正2 正1 1"的正例数据样本；[2019/11/20]

### 提交成绩

提交查看得分与排名：[2019/11/14 17:37]
```
52		new		xmxoxo		0.89160		1
```

### 数据预处理

```
cd code
python prePorcess.py
```

