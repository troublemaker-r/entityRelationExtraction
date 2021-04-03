# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random

import numpy as np
import torch
import torch.nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from sklearn.metrics import f1_score, precision_score, recall_score

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSKE2019SequenceLabeling
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_token, token_label):

        self.guid = guid
        self.text_token = text_token
        self.token_label = token_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, token_label_ids, predicate_label_id,):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_label_ids = token_label_ids
        self.predicate_label_id = predicate_label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class SKE_2019_Sequence_labeling_Processor(DataProcessor):
    """Processor for the SKE_2019 data set"""
    # SKE_2019 data from http://lic2019.ccf.org.cn/kg

    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "token_label_and_one_prdicate_out.txt"), encoding='utf-8') as token_label_out_f:
                    token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                    token_label_out_list = [seq.replace("\n", '') for seq in token_label_out_f.readlines()]
                    assert len(token_in_list) == len(token_label_out_list)
                    examples = list(zip(token_in_list, token_label_out_list))   # 将每行token_in和它的label序列对应
                    return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in_and_one_predicate.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")

    def get_token_labels(self):
        """
            B I O等的标签序列 
        """
        BIO_token_labels = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]  #id 0 --> [Paddding]

        return BIO_token_labels

    def get_predicate_labels(self):
        """
            谓词(关系)标签
        """
        return ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限',
                '出品公司', '出版社', '出生地', '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子',
                '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自', '朝代', '歌手', '母亲',
                '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高',
                '连载网站', '邮政编码', '面积', '首都']

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_token = line
                token_label = None
            else:
                text_token = line[0]   # text_token
                token_label = line[1]  # label 序列
            examples.append(
                InputExample(guid=guid, text_token=text_token, token_label=token_label))
        return examples


def convert_examples_to_features(examples, token_label_list, predicate_label_list, max_seq_length, tokenizer):

    # 得到token_label的dict
    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i
    # 得到predicate_label的dict
    predicate_label_map = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_token = example.text_token.split("\t")[0].split(" ")  # 分隔text
        if example.token_label is not None:
            token_label = example.token_label.split("\t")[0].split(" ")  # 分隔token_label("0", "B-SUB"...)
        else:
            token_label = ["O"] * len(text_token)
        assert len(text_token) == len(token_label)

        text_predicate = example.text_token.split("\t")[1]   # 得到predicate(谓词)
        if example.token_label is not None:
            token_predicate = example.token_label.split("\t")[1]  # 得到 token_label 尾的谓词
        else:
            token_predicate = text_predicate
        assert text_predicate == token_predicate
        # 生成token_b
        tokens_b = [text_predicate] * len(text_token)
        predicate_id = predicate_label_map[text_predicate]

        _truncate_seq_pair(text_token, tokens_b, max_seq_length - 3)

        tokens = []
        token_label_ids = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        token_label_ids.append(token_label_map["[CLS]"])

        for token, label in zip(text_token, token_label):
            tokens.append(token)
            segment_ids.append(0)
            token_label_ids.append(token_label_map[label])

        tokens.append("[SEP]")
        segment_ids.append(0)
        token_label_ids.append(token_label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        #bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
        bias = 1   #1-100 dict index not used
        for token in tokens_b:
          input_ids.append(predicate_id + bias) #add  bias for different from word dict
          segment_ids.append(1)
          token_label_ids.append(token_label_map["[category]"])

        input_ids.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0]) #102
        segment_ids.append(1)
        token_label_ids.append(token_label_map["[SEP]"])

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_label_ids.append(0)
            tokens.append("[Padding]")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info('token: %s' % ' '.join(str(x) for x in tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("token_label_ids: %s" % " ".join([str(x) for x in token_label_ids]))
            logger.info("predicate_id: %s" % str(predicate_id))
        features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    token_label_ids=token_label_ids,
                    predicate_label_id=[predicate_id],
                    ))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def compute_metrics(token_label_ids, token_prediction, token_label_list):
    """
        计算准确率，召回率，F1值 
    """
    token_list_index = [token_label_list.index(value) for value in token_label_list[4:-1]]  # do not care "O"

    token_label_ids = np.reshape(token_label_ids.cpu().numpy(), [-1])   # (1024,)
    token_prediction = np.reshape(token_prediction.cpu().numpy(), [-1])  # (1024,)

    token_label_precision = precision_score(y_true=token_label_ids, y_pred=token_prediction, labels=token_list_index, average="micro")
    token_label_recall = recall_score(y_true=token_label_ids, y_pred=token_prediction, labels=token_list_index, average="micro")
    token_label_f1 = f1_score(y_true=token_label_ids, y_pred=token_prediction, labels=token_list_index, average="micro")

    logger.info("\n")
    logger.info("  Precision score  = %s", token_label_precision)
    logger.info("  Recall score  = %s", token_label_recall)
    logger.info("  F1 score  = %s", token_label_f1)

    return token_prediction, token_label_precision, token_label_recall, token_label_f1


def predicate_id2label(predicate_label_id2label, predicate_prediction):
    """ 将预测的关系id，转为关系name"""
    predicate_prediction_list = []
    for id in predicate_prediction.tolist():
        predicate_label = predicate_label_id2label[id]
        predicate_prediction_list.append(predicate_label)
    return predicate_prediction_list


def token_d2label(token_label_id2label, token_prediction):
    """将预测是token id，转为token label"""
    token_prediction_list = []
    for id in token_prediction:
        token_label = token_label_id2label[id]
        token_prediction_list.append(token_label)
    return token_prediction_list


def train(args, device, train_examples, model, tokenizer, num_train_optimization_steps, token_label_list, predicate_label_list):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    ### 训练和保存模型
    global_step = 0
    train_features = convert_examples_to_features(
        train_examples, token_label_list, predicate_label_list, args.max_seq_length, tokenizer)

    logger.info("********** Running training ****************")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_token_label_ids = torch.tensor([f.token_label_ids for f in train_features], dtype=torch.long)
    all_predicate_label_ids = torch.tensor([f.predicate_label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_token_label_ids,
                               all_predicate_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, token_label_ids, predicate_label_id = batch

            predicate_logits, token_logits, predicate_loss, token_loss \
                = model(input_ids, segment_ids, input_mask, predicate_label_id, token_label_ids)
            # Finally loss
            final_loss = 0.5 * predicate_loss + token_loss
            final_loss.backward()
            loss = final_loss.item()

            logger.info("\n")
            logger.info("global_step = %d", global_step)
            logger.info("loss = %f", loss)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

    # 保存模型，参数，tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_path, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_path, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_path)
    return


def main():
    parser = argparse.ArgumentParser()

    ## 文件参数
    parser.add_argument("--data_dir",
                        default="bin/subject_object_labeling/sequence_labeling_data",
                        type=str,
                        help="训练数据文件夹路径.")
    parser.add_argument("--output_path",
                        default=r"./output/sequnce_labeling_model/",
                        type=str,
                        help="结果输出路径")
    parser.add_argument("--pretrained_dir",
                        default=r"./model/",
                        type=str,
                        help="预训练模型路径")
    ## 模型
    parser.add_argument("--task_name",
                        default="SKE_2019",
                        type=str,
                        help="The name of the task to train.")
    ## 任务选择
    parser.add_argument("--do_train",
                        default=True,
                        type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        default=False,
                        type=bool,
                        help="Whether to run on the test set.")
    ## 运行参数
    parser.add_argument("--max_seq_length",
                        default=480,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        type=bool,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=10,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--predict_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for predict.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    processors = {"ske_2019": SKE_2019_Sequence_labeling_Processor}

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("可使用的cuda数量", n_gpu)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    ##
    ### 验证是否同时不需要train和evaluate，还有在input的目录下是否有训练所需的数据
    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_predict`must be True.")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    ##
    ### 预处理数据集
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    token_label_list = processor.get_token_labels()
    predicate_label_list = processor.get_predicate_labels()

    num_token_labels = len(token_label_list)
    num_predicate_labels = len(predicate_label_list)

    token_label_id2label = {}  # id到label集的映射
    for (i, label) in enumerate(token_label_list):
        token_label_id2label[i] = label
    predicate_label_id2label = {}  # id到关系集的映射
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_id2label[i] = label

    ### 训练阶段
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir, do_lower_case=args.do_lower_case)
        # Prepare model 用已经下载好模型
        model = BertForSKE2019SequenceLabeling.from_pretrained(args.pretrained_dir,
                                                               num_predicate_labels=num_predicate_labels,
                                                               num_token_labels=num_token_labels)
        model.to(device)
        train(args, device, train_examples, model, tokenizer, num_train_optimization_steps, token_label_list, predicate_label_list)

    ### 验证阶段
    if args.do_eval:
        tokenizer = BertTokenizer.from_pretrained(args.output_path, do_lower_case=args.do_lower_case)
        # trained_model_dir = r"pretrained_model\bert-base-chinese"  # 使用预训练模型进行测试
        model = BertForSKE2019SequenceLabeling.from_pretrained(args.output_path,
                                                               num_predicate_labels=num_predicate_labels,
                                                               num_token_labels=num_token_labels)
        model.to(device)
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, token_label_list, predicate_label_list, args.max_seq_length, tokenizer)

        logger.info("********** Running evaluation ***********")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_token_label_ids = torch.tensor([f.token_label_ids for f in eval_features], dtype=torch.long)
        all_predicate_label_ids = torch.tensor([f.predicate_label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_token_label_ids, all_predicate_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        eval_loss, nb_eval_steps = 0, 0
        Predicate_loss,Token_loss,Predicate_prediction, Token_prediction = 0, 0, [], []
        Token_label_precision, Token_label_recall, Token_label_f1 = 0, 0, 0
        result = {}
        for input_ids, input_mask, segment_ids, token_label_ids, predicate_label_id in tqdm(eval_dataloader, desc="Evaluating"):

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            token_label_ids = token_label_ids.to(device)
            predicate_label_id = predicate_label_id.to(device)

            with torch.no_grad():
                predicate_logits, token_logits, predicate_loss, token_loss \
                    = model(input_ids, segment_ids, input_mask, predicate_label_id, token_label_ids)

            softmax = torch.nn.Softmax(dim=-1)
            predicate_probabilities = softmax(predicate_logits)
            predicate_prediction = torch.argmax(predicate_probabilities, dim=-1)

            token_label_probabilities = softmax(token_logits)  # torch.Size([btach_size, 128,10]
            token_predictions = torch.argmax(token_label_probabilities, dim=-1)  # torch.Size([btach_size, 128])
            token_predictions = token_predictions.type_as(token_label_probabilities)
            # Finally loss
            final_loss = 0.5 * predicate_loss + token_loss
            eval_loss += final_loss.item()
            # 计算指标
            token_prediction, token_label_precision, token_label_recall, token_label_f1 = compute_metrics(token_label_ids, token_predictions, token_label_list)

            Predicate_loss += predicate_loss.item()
            Token_loss += token_loss.item()
            predicate_prediction_list = predicate_id2label(predicate_label_id2label, predicate_prediction.cpu().numpy())
            Predicate_prediction.append(predicate_prediction_list)
            token_prediction_list = token_d2label(token_label_id2label, token_prediction)
            Token_prediction .append(token_prediction_list)
            Token_label_precision += token_label_precision
            Token_label_recall += token_label_recall
            Token_label_f1 += token_label_f1
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        Predicate_loss /= nb_eval_steps
        Token_loss /= nb_eval_steps
        Token_label_precision /= nb_eval_steps
        Token_label_recall /= nb_eval_steps
        Token_label_f1 /= nb_eval_steps

        result['eval_loss'] = eval_loss
        result['Predicate_loss'] = Predicate_loss
        result['Token_loss'] = Token_loss
        result['Token_label_precision'] = Token_label_precision
        result['Token_label_recall'] = Token_label_recall
        result['Token_label_f1'] = Token_label_f1
        result['Predicate_prediction'] = Predicate_prediction
        result['Token_prediction'] = Token_prediction

        output_eval_file = os.path.join(args.output_path, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
    ### 预测阶段
    if args.do_predict:
        tokenizer = BertTokenizer.from_pretrained(args.output_path, do_lower_case=args.do_lower_case)
        model = BertForSKE2019SequenceLabeling.from_pretrained(args.output_path,
                                                               num_predicate_labels=num_predicate_labels,
                                                               num_token_labels=num_token_labels)
        model.to(device)
        test_examples = processor.get_test_examples(args.data_dir)
        num_actual_test_examples = len(test_examples)
        test_features = convert_examples_to_features(test_examples, token_label_list, predicate_label_list, args.max_seq_length, tokenizer)

        logger.info("**************** Running Test ******************")
        logger.info("  Num examples = %d", num_actual_test_examples)
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_token_label_ids = torch.tensor([f.token_label_ids for f in test_features], dtype=torch.long)
        all_predicate_label_ids = torch.tensor([f.predicate_label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_token_label_ids, all_predicate_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)
        model.eval()
        result = []

        for input_ids, input_mask, segment_ids, token_label_ids, predicate_label_id in tqdm(test_dataloader, desc="Test"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            token_label_ids = token_label_ids.to(device)
            predicate_label_id = predicate_label_id.to(device)

            with torch.no_grad():
                predicate_logits, token_logits, predicate_loss, token_loss \
                    = model(input_ids, segment_ids, input_mask, predicate_label_id, token_label_ids)

            softmax = torch.nn.Softmax(dim=-1)
            predicate_probabilities = softmax(predicate_logits)
            predicate_prediction = torch.argmax(predicate_probabilities, dim=-1)

            token_label_probabilities = softmax(token_logits)  # torch.Size([btach_size, 128,10]
            token_predictions = torch.argmax(token_label_probabilities, dim=-1)  # torch.Size([btach_size, 128])
            token_predictions = token_predictions.type_as(token_label_probabilities)
            test_dic = {}
            test_dic["token_prediction"] = token_predictions
            test_dic["predicate_probabilities"] = predicate_probabilities
            test_dic["predicate_prediction"] = predicate_prediction
            result.append(test_dic)

        write_path = "./output/sequnce_infer_out/"
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        token_label_output_predict_file = os.path.join(write_path, "token_label_predictions.txt")
        predicate_output_predict_file = os.path.join(write_path, "predicate_predict.txt")
        predicate_output_probabilities_file = os.path.join(write_path, "predicate_probabilities.txt")
        with open(token_label_output_predict_file, "w", encoding='utf-8') as token_label_writer:
            with open(predicate_output_predict_file, "w", encoding='utf-8') as predicate_predict_writer:
                with open(predicate_output_probabilities_file, "w", encoding='utf-8') as predicate_probabilities_writer:
                    num_written_lines = 0
                    logger.info("********* 写入关系预测和token预测的结果 **********")
                    for (i, prediction) in enumerate(result):
                        token_prediction = prediction["token_prediction"].cpu().numpy().tolist()  #(8,128)
                        predicate_probabilities = prediction["predicate_probabilities"].cpu().numpy().tolist() #(8,49)
                        predicate_prediction = prediction["predicate_prediction"].cpu().numpy().tolist()
                        if i >= num_actual_test_examples:
                            break
                        for batch in token_prediction:
                            token_label_output_line = " ".join(token_label_id2label[int(id)] for id in batch)
                            token_label_writer.write(token_label_output_line + "\n")

                        for batch in predicate_probabilities:
                            predicate_probabilities_line = " ".join(str(sigmoid_logit) for sigmoid_logit in batch)
                            predicate_probabilities_writer.write(predicate_probabilities_line + "\n")
                        for id in predicate_prediction:
                            predicate_predict_line = str(predicate_label_id2label[int(id)])
                            predicate_predict_writer.write(predicate_predict_line + "\n")
                            num_written_lines += 1
        logger.info("********* 写入完成 **********")
        assert num_written_lines == num_actual_test_examples

if __name__ == "__main__":
    main()
