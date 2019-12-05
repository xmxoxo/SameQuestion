# 基于Adversarial Attack的问题等价性判别比赛

项目名称: SameQuestion

## 比赛背景

地址：https://www.biendata.com/competition/2019diac/

虽然近年来智能对话系统取得了长足的进展，但是针对专业性较强的问答系统（如法律、政务等），
如何准确的判别用户的输入是否为给定问题的语义等价问法仍然是智能问答系统的关键。
举例而言，“市政府管辖哪些部门？”和“哪些部门受到市政府的管辖？”可以认为是语义上等价的问题，
而“市政府管辖哪些部门？”和“市长管辖哪些部门？”则为不等价的问题。


针对问题等价性判别而言，除去系统的准确性外，系统的鲁棒性也是很重要、但常常被忽略的一点需求。
举例而言，虽然深度神经网络模型在给定的训练集和测试集上常常可以达到满意的准确度，
但是对测试集合的稍微改变（Adversarial Attack）就可能导致整体准确度的大幅度下降
（一些相关文献综述可见https://arxiv.org/pdf/1902.07285.pdf和
https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00254）。
 

如以下样例：

```
	origin example     	     adversarial example
检察机关提起公益诉讼是什么意思	监察机关提起公益诉讼是什么意思
检察机关提起公益诉讼是什么意思	检察机关发起公益诉讼是什么意思
寻衅滋事一般会怎么处理			寻衅兹事一般会怎么处理
什么是公益诉讼					请问什么是公益诉讼 
```

右列数据是左列数据出现一些错别字或者添加一些无意义的干扰词产生的，
并不影响原句的意思，但是这样的微小改动可能会导致完全不同结果。
从用户的角度而言，这意味着稍有不同的输入就可能得到完全不一样的结果，
从而严重降低用户的产品使用体验。

 
【任务提交】


参赛选手需要在验证集和测试集（测试集将于比赛结束前一天发布）上提交预测结果。
提交文件格式参考sample_submission.csv。


注意：请勿改变提交文件的行的顺序。文件每行末尾不要有多余的空格，
包括header line和每一行各列之间；
文件末尾加一个空行；
文件要用无BOM的utf8编码。

-----------------------------------------
## 数据处理思路

配对组合出所有可能，然后用BERT跑一个模型。


```
EquivalenceQuestions=3
NotEquivalenceQuestions=3
组合后应该是：
label=1的6条，昨天的算法
label=0的9条，新组合算法：3*3=9


可西哥  11:11:10

```
fun(lst1=[1,2,3],lst2=[4,5,6])
返回：
1，2，1
1，3，1
2，3，1

1，4，0
1，5，0
1，6，0
2，4，0
2，5，0
2，6，0
3，4，0
3，5，0
3，6，0
```


可西哥  11:11:19
说错了，1的3条

可西哥  11:24:10
有2组

可西哥  11:24:13
EquivalenceQuestions

可西哥  11:24:26
equ的组，表示这些问题都是一样的

NotEquivalenceQuestions
后面三个，表示这些问题跟上面的问题不一样的

现在我们要把这些变成训练的数据
xxx,yyy,1
xxx,yyy,0

一样的就用1表示
不一样就用0表示
```

-----------------------------------------
## QQ群聊天记录

```

夏门-可西哥<xmxoxo@qq.com> 8:59:02
https://www.biendata.com/competition/2019diac/
这个比赛有人感兴趣么，好象就是二分类
【话唠】夏蛋(1134226402) 9:06:42
我怎么觉得这是一个字典解决的事。。。。。
【管理员】夏商周(499244188) 9:09:35
@夏门-可西哥 大佬求带。
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:12:20
字典？@夏蛋
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:12:41
我求大佬组队呢 @夏商周
【话唠】夏蛋(1134226402) 9:12:59
双击查看原图开玩笑的  别介意
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:14:01
没啊，你说的字典是什么思路？
【管理员】夏商周(499244188) 9:14:08
我求抱大佬腿双击查看原图
【话唠】夏蛋(1134226402) 9:14:27
那是工程的方法   略过  纯手工
【话唠】夏蛋(1134226402) 9:14:45
就是构建词库  一对多
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:15:55
官方说不能用手工标注
【话唠】夏蛋(1134226402) 9:16:57
我感觉这个题目  其实就是要对句子意思把握   而且是很细致的那种
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:19:16
嗯
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:19:19
可能 需要 多模型
【群主】猫(997562867) 9:32:14
我觉得这个可以搞啊
【话唠】夏蛋(1134226402) 9:32:35
你开个头
【话唠】夏蛋(1134226402) 9:33:05
拉人，讨论组建起来
【吐槽】Derrick(2901949379) 9:35:57
这个好像之前也有这样的比赛
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:37:13
想组队的私聊我
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:37:25
@Derrick 地址发来看下
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:38:34
要有时间，有项目能力，有一定资源的（能自己跑模型）
【话唠】夏门-可西哥<xmxoxo@qq.com> 9:44:28

【吐槽】Derrick(2901949379) 9:44:39
https://github.com/yanqiangmiffy/sentence-similarity
【群主】猫(997562867) 9:44:50
我最近做的对话模型，正好要处理这类问题
【吐槽】Derrick(2901949379) 9:44:50
就是差不多一个任务
【群主】猫(997562867) 9:45:13
@阿炳 蛋哥，你那个行业数据有不
【吐槽】Derrick(2901949379) 9:45:14
我项目里面添加的语义检索模块也需要处理这个

【群主】猫(997562867)  14:43:27
@夏门-可西哥 我提交在github上了

【话唠】夏门-可西哥(xmxoxo@qq.com)  14:44:49
项目名称是啥？

【群主】猫(997562867)  14:47:39
Chatbot_Retrieval

【话唠】夏门-可西哥(xmxoxo@qq.com)  14:47:58
好的

```



-----------------------------------------
## 项目目录结构

```
X:.				#项目根目录
├─code			#代码
├─data			#数据
│  └─model-01	#模型,多个模型按01,02...依次编号；
├─doc			#参考文档
└─images		#截图
```


-----------------------------------------
## 数据处理 2019/11/14


按一一配对的方式生成全部的训练数据，保存为train_all.tsv 共81869条记录
然后把全数据按8:2拆分成train和dev

训练数据：train.tsv 65553
验证数据：dev.tsv	16388
预测数据：test.tsv	5000

数据预处理：

```
x:
cd X:\project\DC竞赛\基于Adversarial Attack的问题等价性判别比赛\code
preProcess.py
```

处理好的数据在 `/data/`目录下

## 训练模型

训练程序为： run_SameQuestion.py

创建对应的目录:

```
cd /mnt/sda1/transdat/bert-demo/bert/
mkdir ./data/SameQuestion
mkdir ./output/SameQuestion
```

把数据文件拷到服务器,目录为：
`/mnt/sda1/transdat/bert-demo/bert/data/SameQuestion/`


开始训练：
```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion

sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

训练结果：

```
INFO:tensorflow:Finished evaluation at 2019-11-14-08:55:36
INFO:tensorflow:Saving dict for global step 10242: eval_accuracy = 0.9989016, eval_f1 = 0.9975721, eval_loss = 0.0048235836, eval_precision = 0.99918944, eval_recall = 0.9959601, global_step = 10242, loss = 0.0048224577
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 10242: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion/model.ckpt-10242
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.9989016
INFO:tensorflow:  eval_f1 = 0.9975721
INFO:tensorflow:  eval_loss = 0.0048235836
INFO:tensorflow:  eval_precision = 0.99918944
INFO:tensorflow:  eval_recall = 0.9959601
INFO:tensorflow:  global_step = 10242
INFO:tensorflow:  loss = 0.0048224577

```


## 预测结果

使用以下语句来进行预测：

```shell
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

预测结果保存为:
./data/model-01/test_results.tsv

规划：每个模型从序号01开始命名，目录名为：model-序号


## 结果处理

需要把结果处理一下变成可提交的格式：
./code/testProcess.py 01


提交查看得分与排名：[2019/11/14 17:37]
```
52		new		xmxoxo		0.89160		1
```
-----------------------------------------
## 模型提升的思路 2019/11/14

* 数据增强方向：
	同音字替换；

Python拼音转汉字 - WangLinping_CSDN的博客 - CSDN博客  
https://blog.csdn.net/WangLinping_CSDN/article/details/79646347	

	同义词替换；反义词替换；
	回译；文档裁剪
	
	随机drop和shuffle
	code：https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py

数据增强主要采取两种方法,一种是 drop, 对于标题和描述中的字或词,随机的进行删除,用空格代替。
另一种是 shuffle, 即打乱词序。
对于"如何评价 2017 知乎看山杯机器学习比赛?" 这个问题,
使用 drop 对词层面进行处理之后,可能变成"如何 2017 看山杯机器学习 “. 
如果使用 shuffle 进行处理,数据就 可能变成"2017 机器学习?如何比赛知乎评价看山杯”。 

数据 增强对于提升训练数据量,抑制模型过拟合等十分有效.
————————————————
版权声明：本文为CSDN博主「Adupanfei」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Adupanfei/article/details/84956566

NLP数据增强方法 - Adupanfei的博客 - CSDN博客  
https://blog.csdn.net/Adupanfei/article/details/84956566

NLP数据预处理——同义词替换程序 - MrLittleDog的博客 - CSDN博客
https://blog.csdn.net/hfutdog/article/details/81107170

NLP中一些简单的数据增强技术 - Ted_Li - CSDN博客  
https://blog.csdn.net/hero00e/article/details/89523970

tedljw/data_augment: NLP的数据增强Demo  
https://github.com/tedljw/data_augment

数据增强思路： 对已经生成的训练数据做数据增强，这样保持原有处理的流程与逻辑；

* 模型方向：
	语义模型对比；
-----------------------------------------
## 数据增强

参考文章及源码：

NLP中一些简单的数据增强技术 - Ted_Li - CSDN博客  
https://blog.csdn.net/hero00e/article/details/89523970

tedljw/data_augment: NLP的数据增强Demo  
https://github.com/tedljw/data_augment

直接修改上面的源码，对已经生成的训练数据进行二次处理，生成数据增强后的数据:
参数：增强系数=5，变化系数=0.1

```
cd ./code/data_augment
python augment.py --input=../../data/train.tsv --output=train.tsv --num_aug=5 --alpha=0.1
python augment.py --input=../../data/dev.tsv --output=dev.tsv --num_aug=5 --alpha=0.1
```

生成后的数据情况：
训练集: train.tsv	299306
验证集：dev.tsv		74971

创建新模型目录：model-02，训练模型：

开始训练：
```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_L-12_H-768_A-12
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-02

sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

训练结果：

```
INFO:tensorflow:Finished evaluation at 2019-11-15-14:40:17
INFO:tensorflow:Saving dict for global step 46766: eval_accuracy = 0.9897314, eval_f1 = 0.9780534, eval_loss = 0.05940851, eval_precision = 0.9887907, eval_recall = 0.96754676, global_step = 46766, loss = 0.059407715
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 46766: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion-02/model.ckpt-46766
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.9897314
INFO:tensorflow:  eval_f1 = 0.9780534
INFO:tensorflow:  eval_loss = 0.05940851
INFO:tensorflow:  eval_precision = 0.9887907
INFO:tensorflow:  eval_recall = 0.96754676
INFO:tensorflow:  global_step = 46766
INFO:tensorflow:  loss = 0.059407715
```

使用以下语句来进行预测：

```shell
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

结果处理成可提交的格式：

```
./code/testProcess.py 02

```

提交后得分为： 0.8824

-----------------------------------------
## 大模型训练

群里的大佬说要用BERT Large + K fold的方法，可以跑到91以上；

下载了 Large大模型：
chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip
数据来源是：
https://www.ctolib.com/amp/ymcui-Chinese-BERT-wwm.html

训练数据使用最开始生成的标准数据集：

使用大模型进行预测：

```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-03

sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```


运行后OOM了，难道是11G的显卡跑不了Large模型? 报错如下:

```
ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[16,16,128,128] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
         [[node bert/encoder/layer_8/attention/self/Softmax (defined at /mnt/sda1/transdat/bert-demo/bert/modeling.py:722)  = Softmax[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](bert/encoder/layer_8/attention/self/add)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.

         [[{{node loss/Mean/_9471}} = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_6811_loss/Mean", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
```

把`--train_batch_size`参数减小到4，可以正常训练了！
为了快速验证结果，把`--num_train_epochs`也降到了2

```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-03

sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```


运行结果：

```
INFO:tensorflow:Finished evaluation at 2019-11-18-04:08:29
INFO:tensorflow:Saving dict for global step 32776: eval_accuracy = 0.98913836, eval_f1 = 0.97594595, eval_loss = 0.06379507, eval_precision = 0.97938704, eval_recall = 0.97252893, global_step = 32776, loss = 0.06377954
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 32776: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion-03/model.ckpt-32776
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.98913836
INFO:tensorflow:  eval_f1 = 0.97594595
INFO:tensorflow:  eval_loss = 0.06379507
INFO:tensorflow:  eval_precision = 0.97938704
INFO:tensorflow:  eval_recall = 0.97252893
INFO:tensorflow:  global_step = 32776
INFO:tensorflow:  loss = 0.06377954
```

使用以下语句来进行预测：

```shell
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```


生成提交的数据文件：
```
./code/testProcess.py 03
```

提交数据，得分为: 0.8962，比第一次的成绩高了0.5个百分点


改成跑5轮：
```
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME

```

上面的训练相当于在原来的基础上继续再跑5轮了，
训练结果：

```
INFO:tensorflow:Finished evaluation at 2019-11-18-11:04:29
INFO:tensorflow:Saving dict for global step 81941: eval_accuracy = 0.99701, eval_f1 = 0.9933847, eval_loss = 0.022352336, eval_precision = 0.9959394, eval_recall = 0.990843, global_step = 81941, loss = 0.022346897
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 81941: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion-03/model.ckpt-81941
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.99701
INFO:tensorflow:  eval_f1 = 0.9933847
INFO:tensorflow:  eval_loss = 0.022352336
INFO:tensorflow:  eval_precision = 0.9959394
INFO:tensorflow:  eval_recall = 0.990843
INFO:tensorflow:  global_step = 81941
INFO:tensorflow:  loss = 0.022346897

```
生成提交数据，提交验证结果，得分为：`0.8938` 
可见跑5轮过拟合了。

-----------------------------------------
## 数据处理检查及重训练 2019/11/19

队友说群里讨论的数据都是10W+的，而我们的数据只有8.9万，差了比较多，他传了一份上来，记录数是：100298条。
看了一下数据，主要的问题是这样的：
我们的数据生成方式是：
```
正 正 1
正 负 0
```

但这里的“正样本”有分“正１”还是“正２”。
10W+的数据样本是这样生成的：
```
正1 正2 1
正2 正1 1
正 负 0
```

这样正样本的数据量就多出了1万多条。

使用这份数据样本按8:2拆分训练与验证集，条数比例为：80238: 20060，拆分成两个数据文件，然后重新进行训练：
使用filetools工具进行打乱和划分数据；
【注：为了简化数据处理，把数据的header行去除了。】
模型目录名为：model-05

使用large模型，跑3轮：

```shell
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-05

sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

训练结果：
```
INFO:tensorflow:Finished evaluation at 2019-11-19-16:20:09
INFO:tensorflow:Saving dict for global step 60178: eval_accuracy = 0.97926325, eval_f1 = 0.9732304, eval_loss = 0.13369448, eval_precision = 0.96899027, eval_recall = 0.97750777, global_step = 60178, loss = 0.13367452
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 60178: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion-05/model.ckpt-60178
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.97926325
INFO:tensorflow:  eval_f1 = 0.9732304
INFO:tensorflow:  eval_loss = 0.13369448
INFO:tensorflow:  eval_precision = 0.96899027
INFO:tensorflow:  eval_recall = 0.97750777
INFO:tensorflow:  global_step = 60178
INFO:tensorflow:  loss = 0.13367452

```

预测结果：
```
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME

```

生成提交数据，提交得到：0.8992
得分提高了很多，基本上接近0.90了

在这个基础上继续跑5轮看下：
```
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME

```

训练结果：

```
INFO:tensorflow:Finished evaluation at 2019-11-19-20:35:56
INFO:tensorflow:Saving dict for global step 100297: eval_accuracy = 0.9846967, eval_f1 = 0.9801178, eval_loss = 0.11168764, eval_precision = 0.9820896, eval_recall = 0.97815406, global_step = 100297, loss = 0.11167095
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 100297: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion-05/model.ckpt-100297
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.9846967
INFO:tensorflow:  eval_f1 = 0.9801178
INFO:tensorflow:  eval_loss = 0.11168764
INFO:tensorflow:  eval_precision = 0.9820896
INFO:tensorflow:  eval_recall = 0.97815406
INFO:tensorflow:  global_step = 100297
INFO:tensorflow:  loss = 0.11167095
```

预测：

```
sudo python run_SameQuestion.py \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME

```

生成提交格式数据，提交结果。得分为：0.898
看来跑太多轮是会过拟合的。
得分最高的结果0.8992中，最前面几个数据预测值是这样的：
```
0	0
1	0
2	1
3	0
4	1
5	1
6	0
7	1
8	1
9	0
10	0
11	1
```


-----------------------------------------
## 分支思路

思路：负样本是否也可以对调来扩大数据量？

```
reduction(2233432320)  9:17:47
这个得看模型了。如果是通过向量叠加方式计算的就有效果，如果是量化为向量距离那就没效果

```

同时在另一张显卡上同时开始训练另一个模型？

最优的轮数是否为2？


设置第1张显卡进行训练模型,可以使用以下方式指定，命令行运行：
```
export CUDA_VISIBLE_DEVICES=1
```


使用：负 正 0 的模式扩展数据
训练轮数设置为2，模型目录 ：model-06


修改训练脚本，增加了可指定GPU的参数，默认为0需要时可指定为1，方便同时训练。
由于数据增加，batch size=4也跑不动，降为2可以跑。
```
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-06

sudo python run_SameQuestion.py \
  --GPU=1 \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

训练了10个小时左右，训练结果：

```
INFO:tensorflow:Finished evaluation at 2019-11-20-03:40:33
INFO:tensorflow:Saving dict for global step 130990: eval_accuracy = 0.84658605, eval_f1 = 0.65052867, eval_loss = 0.682613, eval_precision = 0.6969742, eval_recall = 0.6098865, global_step = 130990, loss = 0.68252987
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 130990: /mnt/sda1/transdat/bert-demo/bert/output/SameQuestion-06/model.ckpt-130990
INFO:tensorflow:evaluation_loop marked as finished
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.84658605
INFO:tensorflow:  eval_f1 = 0.65052867
INFO:tensorflow:  eval_loss = 0.682613
INFO:tensorflow:  eval_precision = 0.6969742
INFO:tensorflow:  eval_recall = 0.6098865
INFO:tensorflow:  global_step = 130990
INFO:tensorflow:  loss = 0.68252987
```

预测结果：
```
sudo python run_SameQuestion.py \
  --GPU=1 \
  --task_name=setiment \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER/$EXP_NAME \
  --max_seq_length=128 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

生成提交格式提交结果，得分：0.7364


-----------------------------------------
## KFold 思路与流程 2019/11/20


比如K=5, 数据分成D1-D5,测试数据T

模型1：训练集[D1,D2,D3,D4]，验证集D5，跑出模型M1，预测D5并计算出ACC1，预测T得到结果T1
模型2：训练集[D1,D2,D3,D5]，验证集D4，跑出模型M2，预测D4并计算出ACC2，预测T得到结果T2
模型3：训练集[D1,D2,D4,D5]，验证集D3，跑出模型M3，预测D3并计算出ACC3，预测T得到结果T3
模型4：训练集[D1,D3,D4,D5]，验证集D2，跑出模型M4，预测D2并计算出ACC4，预测T得到结果T4
模型5：训练集[D2,D3,D4,D5]，验证集D1，跑出模型M5，预测D1并计算出ACC5，预测T得到结果T5

最终预测结果：
加权平均：TR = (∑(Ti*ACCi) )/ (∑ACCi)


***数据预处理***

增加了kfold的数据自动生成，指定fold_k参数即可自动生成K份训练数据。
增加把test.tsv测试集自动放入目录中，不需要手工复制。

已经生成了K=5份训练数据，数据自动保存在./data/kfold/目录下。

传到服务器上，目录名称设置为：SameQuestion-07

开始训练这5个模型,为了方便直接把训练，验证，测试全部一次执行完成：

模型编号为奇数使用GPU0，偶数使用GPU1

训练模型1：

```
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-07/model_1

sudo python run_SameQuestion.py \
  --GPU=0 \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

训练模型2
```
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-07/model_2

sudo python run_SameQuestion.py \
  --GPU=1 \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

模型1和模型2在晚上23点左右完成训练，总用时约为6小时。

模型3和模型4，只要修改模型目录名称就可以开始训练了，但是要注意对应的screen：

模型3：
```
export EXP_NAME=SameQuestion-07/model_3
```

模型4：
```
export EXP_NAME=SameQuestion-07/model_4
```

模型5：
```
export EXP_NAME=SameQuestion-07/model_5
```


模型融合的核心算法：

```
acc =[0.9584489,0.9682052,0.9674128,0.96884906]
T1 = [[0.9516069,0.048393086],[0.9516069,0.048393086]]
T2 = [[0.9997036,0.0002963326],[0.9997036,0.0002963326]]
T3 = [[0.9974492,0.0025507766],[0.9974492,0.0025507766]]
T4 = [[0.9990158,0.0009841687],[0.9990158,0.0009841687]]

T = [T1,T2,T3,T4]

import numpy as np
from functools import reduce

def merge(T,acc):
	Tm = map(lambda x,y: np.array(x)*y, T,acc)
	Tr = reduce(lambda x,y: x+y, Tm)
	TR = list(Tr/sum(acc))
	return TR

merge(T,acc)
[0.9870329803139279, 0.012966985608771003]

```

-----------------------------------------
## K-Fold 模型融合 2019/11/22

根据目录结构，编写程序自动对输出结果进行数据处理，计算加权平均的结果。
程序自动遍历指定的目录，然后会

命令行调用, 运行过程与结果：

```
 9:47:37.61|X:>model_merge.py ../data/model-07/
正在自动融合模型结果...
子目录清单:
../data/model-07/model_1
../data/model-07/model_2
../data/model-07/model_3
../data/model-07/model_4
../data/model-07/model_5
------------------------------
ACC: [0.9584489, 0.9682052, 0.9674128, 0.96884906, 0.97107625]
T的大小:5, T[0]的大小：5000
计算后的shape：(5000, 2)
------------------------------
          0         1  result
0  0.989532  0.010468       0
1  0.999531  0.000469       0
2  0.598490  0.401510       0
3  0.996340  0.003659       0
4  0.401169  0.598831       1
5  0.000874  0.999126       1
6  0.996234  0.003766       0
7  0.196693  0.803307       1
8  0.000909  0.999091       1
9  0.880732  0.119268       0
数据已生成:../data/model-07/result.csv

```

提交融合后的结果，得分： 0.90840

```
48	↑43	xmxoxo 0.90840	7
```
-----------------------------------------
## 基于字符特征的数据集训练 2019/11/22

目录名称： model-08
数据：手工按8:2拆分，总记录：100298,拆分后： 80238: 20060

开始训练模型，由于特征字符串很长，参数做了调整：
max_seq_length改为512
num_train_epochs改为2

由于没有测试集，先去掉预测参数 `  --do_predict=true \`

```
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-08

sudo python run_SameQuestion.py \
  --GPU=0 \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

尝试各种参数都是OOM

-----------------------------------------
## 关于“标签泄露”

```
永不息的舞步  10:02:58
那个线上线下差距大是因为标签泄露的原因

可西哥  10:03:20
标签泄露？

可西哥  10:03:29
这个是什么意思？

永不息的舞步  10:03:58
对的，就是验证集有训练集中的训练样例

可西哥  10:04:08
噢

可西哥  10:04:17
但是不可能有完全一样的样本啊

永不息的舞步  10:04:18
我现在搞了一下，线上线下差距0.0003

可西哥  10:04:30
怎么搞的？

永不息的舞步  10:05:03
有的，你验证集句子a或句子b肯定在训练集中了

永不息的舞步  10:05:26
pair<a, b>

可西哥  10:05:34
就是要保证在验证集里的句子a不出现在训练集中？

永不息的舞步  10:05:47
对的，

永不息的舞步  10:06:07
这样就不会标签泄露了，困扰了我一段时间

永不息的舞步 2019/11/27 10:22:14
我担心下游加多了，会破坏预训练权重

可西哥 2019/11/27 10:23:20
应该还好吧，就当BERT做向量了

永不息的舞步  10:39:17
你可以向ESIM方向试试
```
-----------------------------------------
## 新数据训练

目录名：SameQuestion-09	

训练数据：
```
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-09

sudo python run_SameQuestion.py \
  --GPU=1 \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

训练结果：

```
eval_accuracy = 0.88394815
你的提交分数为 0.8756.
```

继续从2轮跑到5轮：



-----------------------------------------
## 新的训练数据

目录名：SameQuestion-10

训练数据：
```
cd /mnt/sda1/transdat/bert-demo/bert/
export BERT_BASE_DIR=/mnt/sda1/transdat/bert-demo/bert/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
export GLUE_DIR=/mnt/sda1/transdat/bert-demo/bert/data
export TRAINED_CLASSIFIER=/mnt/sda1/transdat/bert-demo/bert/output
export EXP_NAME=SameQuestion-10

sudo python run_SameQuestion.py \
  --GPU=0 \
  --task_name=setiment \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$EXP_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$TRAINED_CLASSIFIER/$EXP_NAME
```

本地验证集：0.88354933
提交结果： 0.872

-----------------------------------------
## 语义解析思路

```
借车撞人了，被撞者能找车主去索赔吗?	交通事故未划分责任允许放车吗	0

借车撞人了，被撞者能找车主去索赔吗?	
==> 撞人, 找,索赔
交通事故未划分责任允许放车吗
==>允许,放,车

租赁合同应当以什么形式订立	订立租赁合同的正确形式是什么	1

租赁合同应当以什么形式订立
==>租赁合同,订立,形式

订立租赁合同的正确形式是什么
==>订立, 租赁合同,形式, 是什么


在公司要怎么分配会计人员	小企业应该设哪几个会计岗位？	0

在公司要怎么分配会计人员
==>要,怎么分配, 会计人员

小企业应该设哪几个会计岗位？
==> 小企业， 应该设， 哪几个，会计岗位


公司规定休产假就没有工资，合法吗？	产假期间，用人单位不发工资合法么？	1

公司规定休产假就没有工资，合法吗？
==>没有工资,合法,吗,?

产假期间，用人单位不发工资合法么？
==>不发工资,合法,么？

```


-----------------------------------------
## 比赛群沟通内容 

沐鑫:
还有对付这种不平衡样本，用凯明大神的focal loss
King:
focal loss对不平衡起作用的

感谢大佬们的分享，总结了一下，大家也可以自己增加一些资料或者艾特我来添加
https://github.com/WenRichard/DIAC2019-Adversarial-Attack-Share

focalloss

focal loss论文笔记(附基于keras的多类别focal loss代码) https://blog.csdn.net/qq_42277222/article/details/81711289

https://github.com/maozezhong/focal_loss_multi_class


非平衡数据集 focal loss 多类分类 https://www.zhuanzhi.ai/document/296fe8b5779b6daec7d1caf66d4f105e

https://github.com/clcarwin/focal_loss_pytorch 
https://github.com/ailias/Focal-Loss-implement-on-Tensorflow 
https://github.com/mkocabas/focal-loss-keras

关于比赛的分享：
https://github.com/WenRichard/DIAC2019-Adversarial-Attack-Share



