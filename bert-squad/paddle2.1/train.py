import os
import json
import sys
import time
import argparse
import ctypes
import subprocess

import paddle
from paddle.io import Dataset, DataLoader
from tqdm import *

sys.path.append('..')
from common.squad import (load_squad_features, RawResult, get_answers)
from src.config import config
from src.network import BertForQuestionAnswering, BertConfig
from src.lr_scheduler import LinearWarmUpScheduler

PROF_SAMPLES_PER_EPOCH = 3200

def parse_arg():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--init_checkpoint", default=None, type=str, help="The checkpoint file from pretraining")
    parser.add_argument("--config_file", default=None, type=str, required=False,
                        help="The BERT model config for loading checkpoint file")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--disable-progress-bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument("--skip_cache",
                        default=False,
                        action='store_true',
                        help="Whether to cache train features")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Location to cache train feaures. Will default to the dataset directory")
    parser.add_argument("--is-prof",
                        default=False,
                        action='store_true',
                        help="Whether to profile model.")
    parser.add_argument("--train-batch-size", default=None, type=int,
                        help="The batch size of training.")
    args = parser.parse_args()
    return args


# Because padddle 2.1 does not support TensorDataset on GPU
# We construct the dataset for SQuAD
class SQuADDataset(Dataset):
    def __init__(self, features, is_train=True):
        self.features = features
        self.is_train = is_train

    def __getitem__(self, idx):
        input_ids = paddle.to_tensor(self.features[idx].input_ids, dtype=paddle.int64)
        input_mask = paddle.to_tensor(self.features[idx].input_mask, dtype=paddle.int64)
        segment_ids = paddle.to_tensor(self.features[idx].segment_ids, dtype=paddle.int64)
        if self.is_train: 
            start_position = paddle.to_tensor(self.features[idx].start_position, dtype=paddle.int64)
            end_position = paddle.to_tensor(self.features[idx].end_position, dtype=paddle.int64)
            return input_ids, input_mask, segment_ids, start_position, end_position
        else:
            return input_ids, input_mask, segment_ids, paddle.to_tensor(idx, dtype=paddle.int64)

    def __len__(self):
        return len(self.features)


def main():
    
    args = parse_arg()
    print("---------configurations--------------")
    for k, v in vars(args).items():
        print(k,':',v)
    print("-------------------------------------")
    if args.train_batch_size is not None:
        config['train-batch-size'] = args.train_batch_size

    # make output dir if not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _cuda_tools_ext = ctypes.CDLL("libnvToolsExt.so")


    # model
    model = BertForQuestionAnswering.from_pretrained(args.bert_model)
    print(f"Load model from pretrained: {args.bert_model}")
    for name, param in list(model.named_parameters()):
        mean = param.mean().item()
        var = param.var().item()
        print(f"{name}: {mean:.6f}, {var:.6f}")

    # Train DataLoader
    _, train_features = load_squad_features(args, config["train-file"], True)
    train_data = SQuADDataset(train_features, True)
    train_dataloader = DataLoader(train_data, batch_size=config["train-batch-size"], shuffle=True,
                                num_workers=config['dataset-num-workers'])

    step_size = int(len(train_features) / config["train-batch-size"])
    num_train_optimization_steps = step_size * args.num_train_epochs

    # Eval DataLoader
    eval_examples, eval_features = load_squad_features(args, config["predict-file"], False)
    eval_data = SQuADDataset(eval_features, False)
    eval_dataloader = DataLoader(eval_data, batch_size=config["predict-batch-size"], shuffle=False,
                                num_workers=config['dataset-num-workers'])

    # LR Scheduler
    lr = config['lr-base'] * config["train-batch-size"] / config['lr-batch-denom']
    scheduler = LinearWarmUpScheduler(lr, warmup=config['lr-warmup-proportion'], total_steps=num_train_optimization_steps)

    # Optimizer
    assert config['optimizer-type'] == 'AdamW'
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n[1] for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer = paddle.optimizer.AdamW(parameters=param_optimizer, learning_rate=scheduler, beta1=0.9, beta2=0.999,
                                        epsilon=1e-6, weight_decay=config['weight-decay'],
                                        apply_decay_param_fun=(lambda x: not any(nd in x for nd in no_decay)))
    
    # Record the Total time for train
    total_train_time = 0
    for epoch in range(int(args.num_train_epochs)):
        # Log some infomations
        print(f"--------Epoch: {epoch:03}, " +
              f"lr: {optimizer._learning_rate():f}--------")
        
        model.train()
        train_iter = tqdm(train_dataloader(), desc="Iteration", disable=args.disable_progress_bar)

        # Training loop
        start_time = time.time()
        _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"epoch:{epoch}".encode('utf-8')))
        for num_steps, batch in enumerate(train_iter):
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            # Compute prediction and loss
            start_logits, end_logits = model(input_ids, segment_ids, input_mask)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions.clip_(0, ignored_index)
            end_positions.clip_(0, ignored_index)
            loss_fct = paddle.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
            # Backpropagation
            optimizer.clear_grad()
            loss.backward()
            # Update (TODO: gradient clipping max_grad_norm=1.0)
            optimizer.step()
            scheduler.step()
            if args.is_prof:
                through_samples =  (num_steps + 1) * config["train-batch-size"]
                if through_samples >= PROF_SAMPLES_PER_EPOCH:
                    break

        final_loss = loss.item()
        _cuda_tools_ext.nvtxRangePop()
        total_train_time += time.time() - start_time
        print(f"Train: Loss(last step): {final_loss:>.4e}, " + 
            f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")

        if not args.is_prof:
            # Validation process
            model.eval()
            all_results = []
            eval_iter = tqdm(eval_dataloader(), desc="Iteration", disable=args.disable_progress_bar)
            for batch in eval_iter:
                input_ids, input_mask, segment_ids, example_indices = batch
                # Forward computing
                with paddle.no_grad():
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].numpy().tolist()
                    end_logits = batch_end_logits[i].numpy().tolist()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                            start_logits=start_logits,
                                            end_logits=end_logits))
            
            # Write results into output file
            answers, _ = get_answers(eval_examples, eval_features, all_results, args)
            output_prediction_file = args.output_dir + "/predictions.json"
            with open(output_prediction_file, "w") as f:
                f.write(json.dumps(answers, indent=4) + "\n")

            # Running the eval script
            eval_script = os.path.join(os.path.dirname(config["predict-file"]), config["eval_script"])
            print('script is {}'.format(eval_script))
            eval_out = subprocess.check_output([sys.executable, eval_script,
                                                config["predict-file"], output_prediction_file])
            scores = str(eval_out).strip()
            exact_match = float(scores.split(":")[1].split(",")[0])
            f1 = float(scores.split(":")[2].split(",")[0])
            print(f"Test: exact_match: {exact_match}, F1: {f1}")
        
    # Print the training time
    print("Time used: %.2fs" % (total_train_time))

if __name__ == "__main__":
    main()
