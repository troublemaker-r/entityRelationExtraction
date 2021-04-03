from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import Args

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from sklearn.metrics import f1_score,precision_score,recall_score


from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForMultiSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam



logger = logging.getLogger(__name__)

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class PaddingInputExample(object):
    '''

    '''

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example

class DataProcessor(object):

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
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class ZYProcessor(DataProcessor):

    def __init__(self):
        self.language = 'zh'

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, 'token_in.txt'),encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "predicate_out.txt"), encoding='utf-8') as predicate_out_f:
                token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                predicate_label_list = [seq.replace("\n", '') for seq in predicate_out_f.readlines()]
                assert len(token_in_list) == len(predicate_label_list)
                examples = list(zip(token_in_list, predicate_label_list))
                return examples

    def get_train_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "train")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(self.get_examples(os.path.join(data_dir, "valid")), "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")

    def get_labels(self):
        return ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
                '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
                '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
                '邮政编码', '面积', '首都']

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_str = line
                predicate_label_str = '丈夫'
            else:
                text_str = line[0]
                predicate_label_str = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_str, text_b=None, label=predicate_label_str))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):

    label_map = {}
    for (i,label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        token_a = example.text_a.split(' ')
        token_b = None

        if len(token_a) > max_seq_length - 2 :
            token_a = token_a[0 : max_seq_length - 2]

        tokens = []
        segment_ids = []

        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in token_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_list = example.label.split(' ')
        label_ids = _predicate_label_to_id(label_list,label_map)

        if ex_index < 3 :
            logger.info('***实例***')
            logger.info('guid: %s' % (ex_index))
            logger.info('token: %s' % ' '.join(str(x) for x in tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            is_real_example=True
        )
        features.append(feature)
    return features

def _predicate_label_to_id(predicate_label, predicate_label_map):
    predicate_label_map_length = len(predicate_label_map)
    predicate_label_ids = [0] * predicate_label_map_length
    for label in predicate_label:
        predicate_label_ids[predicate_label_map[label]] = 1
    return predicate_label_ids

def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def main():
    parser = Args.Parser().getParser()
    args = parser.parse_args()

    processors = {'zy': ZYProcessor,}

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO )

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_predict` must be True.")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size) * args.num_train_epochs

    model = BertForMultiSequenceClassification.from_pretrained(args.model_dir,
              num_labels=num_labels)

    model.to(device)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float32)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                logits , _ = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits , label_ids)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                logger.info('')
                logger.info('step: %d' %(step))
                logger.info('loss: %f' %(loss.item()))


        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

    if args.do_eval:
        model = BertForMultiSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float32)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()

        recall = 0
        precision = 0
        F1 = 0
        steps = 0
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits, probilities = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, label_ids)

            pred = probilities.detach().cpu().numpy()
            label = label_ids.detach().cpu().numpy()
            pred[pred > 0.5] = 1.0
            pred[pred <= 0.5] = 0.0

            steps += 1
            F1 += f1_score(y_true=label, y_pred=pred, average='micro')
            precision += precision_score(y_true=label, y_pred=pred, average='micro')
            recall += recall_score(y_true=label, y_pred=pred, average='micro')

            logger.info('')
            logger.info('loss: %f' %(loss.item()))
            logger.info('recall: %f' %(recall/steps))
            logger.info('precision %f' %(precision/steps))
            logger.info('f1: %f' %(F1/steps))

    if args.do_predict:
        model = BertForMultiSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        pred_examples = processor.get_test_examples(args.data_dir)
        pred_features = convert_examples_to_features(
            pred_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(pred_examples))
        logger.info("  Batch size = %d", args.predict_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in pred_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in pred_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in pred_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in pred_features], dtype=torch.float32)

        pred_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        pred_sampler = SequentialSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=args.predict_batch_size)

        model.eval()

        preds = []
        for input_ids, input_mask, segment_ids, label_ids in tqdm(pred_dataloader, desc="predicting"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                _, probilities = model(input_ids, segment_ids, input_mask, labels=None)

            pred = probilities.detach().cpu().numpy()
            pred = pred.tolist()
            preds.append(pred)

        write_path = "./output/predicate_infer_out/"
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        score_file = "./output/predicate_infer_out/probility.txt"
        predict_file = "./output/predicate_infer_out/predicate_predict.txt"

        logger.info('')
        logger.info('***********writing predict result**********')
        with open(score_file, 'w', encoding='utf-8') as score_writer:
            with open(predict_file, 'w', encoding='utf-8') as predict_writer:
                num_total_lines = 0
                for batch in preds:
                    for lines in batch:
                        score = ' '.join(str(number) for number in lines)+'\n'
                        score_writer.write(score)
                        predict_relation = []
                        for idx,prob in enumerate(lines):
                            if prob > 0.5:
                                predict_relation.append(label_list[idx])
                        predict = ' '.join(predict_relation) + '\n'
                        predict_writer.write(predict)
                        num_total_lines += 1
        assert num_total_lines == len(pred_examples)

main()