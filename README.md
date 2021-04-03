**:mag_right: 原项目地址（`Tensorflow`实现）**：[https://github.com/yuanxiaosc/Entity-Relation-Extraction](https://github.com/yuanxiaosc/Entity-Relation-Extraction)

### :yum: 1. 项目简介

#### 任务：

​	1.管道式三元组抽取模型。第一层模型为多标签关系分类模型，第二层模型为基于上层模型输出关系的序列标注模型。
​	2.两个模型可以分开训练和验证，但是预测三元组时要按顺序执行。
​	3.数据为百度2019语言与智能技术竞赛中的数据。

#### 数据

- 原始数据存放于raw_data文件夹下。
  - 训练集：train_data.json，测试集：dev_data.json
- 文本内容： "text": "内容简介《宜兴紫砂图典》由故宫出版社出版"
- 三元组标注： "spo_list": [{"predicate": "出版社", "object_type": "出版社", "subject_type": "书籍", "object": "故宫出版社", "subject": "宜兴紫砂图典"}]}
  - 注明： 数据预处理脚本中的竞赛模式（Competition_Mode）设置为False时会将验证集作为测试集转化（便于查看效果），设置为True时会转化测试集数据。	

### :yum: 2. 预训练模型：

​	存放在model文件夹下，需要有以下三个文件

- `bert_config.json`：bert的模型的基本配置文件
- `pytorch_model.bin`：预训练模型
- `vocab.txt`：字典文件

### :yum: ​3. 数据预处理：

- 准备关系数据：运行 `bin/predicate_classifiction `文件夹下的 `predicate_data_manager.py`脚本，之后会在当前目录下生成训练，验证和测试数据集.
- 准备标注数据：运行` bin/subject_object_labeling `文件夹下的 `sequence_labeling_data_manager.py`脚本，之后会在当前目录下生成训练，验证和测试数据集

### :yum: 4. 参数简介：

​	**关系分类**

``` text
	--data_dir ：数据预处理后的数据路径
	--model_dir： 预训练模型路径
	--task_name： 任务名
    --vocab_file： 预训练词表位置
	--output_dir： 输出保存路径
	--do_lower_case： 英文字母是否转化为小写
	--max_seq_length： 最大句长
	--do_train： 是否训练
	--do_eval： 是否验证
	--do_predict： 是否预测
	--train_batch_size： 训练批次大小
	--eval_batch_size： 验证批次大小
	--predict_batch_size： 预测批次大小
	--learning_rate： 学习率
	--num_train_epochs： 一共训练几个迭代
	--warmup_proportion： 热身学习占比
```

​	**序列标注**

```text
## 文件参数
	--data_dir ：数据预处理后的数据文件路径
	--output_path： 模型和验证结果输出路径
	--pretrained_dir： 预训练模型路径
## 模型
	--task_name： 任务名
## 任务选择
	--do_train： 是否训练
	--do_eval： 是否验证
	--do_predict： 是否预测
## 运行参数
	--max_seq_length： 最大句长
	--train_batch_size： 训练批次大小
	--eval_batch_size： 验证批次大小
	--predict_batch_size： 预测批次大小
	--learning_rate： 	 学习率
	--num_train_epochs：  反复训练几次
	--warmup_proportion： 热身学习占比
```
### :yum: 5. 模型使用（训练和验证可以同时进行）：

- 训练和验证时： 
  - 关系分类模型：运行 `relation_classifier.py` 脚本，参数可在`Args.py`脚本中调整
  - 序列标注模型：运行` run_sequnce.py` 脚本，参数在主函数代码中调整。

```text
预测时，先运行 relation_classifier.py 脚本，然后在 bin/predicate_classifiction 文件夹下运行 prepare_data_for_labeling_infer.py 脚本生成关系预测的结果。
然后再运行 run_sequnce.py 脚本，生成最终的标注结果。最后运行 produce_submit_json_file.py 脚本，生成三元组抽取结果。

流程结束后在output文件夹下，生成以下五个文件夹：
	predicate_infer_out 存放关系分类预测的结果（用于序列标注）
	sequence_infer_out 存放序列标注预测的结果
	sequnce_labeling_model 存放序列标注模型
	predicate_classifiction_model 存放关系分类的模型
```

- 三元组抽取输出：
  - 抽取结果在`output/final_text_spo_list_result`文件夹下，同原始数据的标注类似（如下）。

 ~~~text
"text": "如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈"
"spo_list": [{"object_type": "人物", "predicate": "主演", "object": "周星驰",
"subject_type": "影视作品", "subject": "喜剧之王"}]
 ~~~

