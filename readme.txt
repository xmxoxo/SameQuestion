#基于Adversarial Attack的问题等价性判别比赛

项目名称: SameQuestion

地址：
https://www.biendata.com/competition/2019diac/
-----------------------------------------
##比赛背景
 

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

          origin example     	     adversarial example
检察机关提起公益诉讼是什么意思	监察机关提起公益诉讼是什么意思
检察机关提起公益诉讼是什么意思	检察机关发起公益诉讼是什么意思
寻衅滋事一般会怎么处理			寻衅兹事一般会怎么处理
什么是公益诉讼					请问什么是公益诉讼 

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
## 思路

配对组合出所有可能，然后用BERT跑一个模型。

EquivalenceQuestions=3
NotEquivalenceQuestions=3
组合后应该是：
label=1的6条，昨天的算法
label=0的9条，新组合算法：3*3=9


可西哥  11:11:10
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
-----------------------------------------
## QQ群聊天记录


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
-----------------------------------------
## 项目目录结构

X:.				#项目根目录
├─code			#代码
├─data			#数据
│  └─model-01	#模型,多个模型按01,02...依次编号；
├─doc			#参考文档
└─images		#截图


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
生成提交数据，提交验证结果，得分为：`0.8938` 可见跑5轮并没有提高多少得分。

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



