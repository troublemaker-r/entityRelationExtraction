import argparse

class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()
        # Required parameters
        parser.add_argument('--data_dir',
                            type=str,
                            default='./bin/predicate_classifiction/classification_data',
                            help='The input data dir. Should contain the data files.')
        parser.add_argument('--model_dir',
                            type=str,
                            default='./model',
                            help='The config json file and model file corresponding to the pre-trained BERT model.')
        parser.add_argument('--task_name',
                            type=str,
                            default='zy',
                            help='The name of the task to train.')
        parser.add_argument('--vocab_file',
                            type=str,
                            default='./model/vocab.txt',
                            help='The vocabulary file that the BERT model was trained on.')
        parser.add_argument('--output_dir',
                            type=str,
                            default='./output/predicate_classifiction_model',
                            help="The output directory where the model checkpoints will be written.")

        # Other parameters
        parser.add_argument('--do_lower_case',
                            type=bool,
                            default=True,
                            help="Whether to lower case the input text."
                                 "Should be True for uncased models and False for cased models.")
        parser.add_argument('--max_seq_length',
                            type=int,
                            default=128,
                            help="The maximum total input sequence length after WordPiece tokenization."
                                 "Sequences longer than this will be truncated, and sequences shorter "
                                 "than this will be padded.")
        parser.add_argument('--do_train',
                            type=bool,
                            default=False,
                            help="Whether to run training.")
        parser.add_argument('--do_eval',
                            type=bool,
                            default=False,
                            help="Whether to run eval on dev set.")
        parser.add_argument('--do_predict',
                            type=bool,
                            default=True,
                            help="Whether to run the model in inference mode on the test set.")
        parser.add_argument('--train_batch_size',
                            type=int,
                            default=32,
                            help="Total batch size for training.")
        parser.add_argument('--eval_batch_size',
                            type=int,
                            default=8,
                            help="Total batch size for eval.")
        parser.add_argument('--predict_batch_size',
                            type=int,
                            default=8,
                            help="Total batch size for predict.")
        parser.add_argument('--learning_rate',
                            type=float,
                            default=2e-5,
                            help="The initial learning rate for Adam.")
        parser.add_argument('--num_train_epochs',
                            type=float,
                            default=6.0,
                            help="Total number of training epochs to perform.")
        parser.add_argument('--warmup_proportion',
                            type=float,
                            default=0.1,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10% of training.")
        return parser