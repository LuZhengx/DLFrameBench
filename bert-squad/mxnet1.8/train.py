import os
import json
import sys
import time
import argparse
import ctypes
import subprocess

os.environ["MXNET_USE_FUSION"] = "0"
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

import mxnet
from mxnet.gluon.data import DataLoader, ArrayDataset
from tqdm import *

sys.path.append('..')
from common.squad import (load_squad_features, RawResult, get_answers)
from src.config import config
from src.network import BertForQuestionAnswering, BertConfig
from src.lr_scheduler import LinearWarmUpScheduler
from src.optimizer import AdamW


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
    parser.add_argument("--is_prof",
                        default=False,
                        action='store_true',
                        help="Whether to profile model.")
    args = parser.parse_args()
    return args


def main():
    
    args = parse_arg()
    print("---------configurations--------------")
    for k, v in vars(args).items():
        print(k,':',v)
    print("-------------------------------------")

    # make output dir if not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    _cuda_tools_ext = ctypes.CDLL("libnvToolsExt.so")

    # model
    model = BertForQuestionAnswering.from_pretrained(args.bert_model, args.max_seq_length)
    model.hybridize()
    print(f"Load model from pretrained: {args.bert_model}")
    for name, param in list(model.collect_params().items()):
        mean = param.data().mean().asscalar()
        var = ((param.data() - mean)**2).sum().asscalar() / param.data().size
        print(f"{name}: {mean:.6f}, {var:.6f}, {param.shape}")

    # Train DataLoader
    _, train_features = load_squad_features(args, config["train-file"], True)
    all_input_ids = mxnet.nd.array([f.input_ids for f in train_features], dtype='int64')
    all_input_mask = mxnet.nd.array([f.input_mask for f in train_features], dtype='int64')
    all_segment_ids = mxnet.nd.array([f.segment_ids for f in train_features], dtype='int64')
    all_start_positions = mxnet.nd.array([f.start_position for f in train_features], dtype='int64')
    all_end_positions = mxnet.nd.array([f.end_position for f in train_features], dtype='int64')
    train_data = ArrayDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions)
    train_dataloader = DataLoader(train_data, batch_size=config["train-batch-size"], shuffle=True,
                                  last_batch='keep', num_workers=config['dataset-num-workers'])

    step_size = int(len(train_features) / config["train-batch-size"])
    num_train_optimization_steps = step_size * args.num_train_epochs

    # Eval DataLoader
    eval_examples, eval_features = load_squad_features(args, config["predict-file"], False)
    all_input_ids = mxnet.nd.array([f.input_ids for f in eval_features], dtype='int64')
    all_input_mask = mxnet.nd.array([f.input_mask for f in eval_features], dtype='int64')
    all_segment_ids = mxnet.nd.array([f.segment_ids for f in eval_features], dtype='int64')
    all_example_index = mxnet.nd.arange(all_input_ids.shape[0], dtype='int64')
    eval_data = ArrayDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=config["predict-batch-size"], shuffle=False,
                                 last_batch='keep', num_workers=config['dataset-num-workers'])
    
    # LR Scheduler
    lr = config['lr-base'] * config["train-batch-size"] / config['lr-batch-denom']
    scheduler = LinearWarmUpScheduler(lr, warmup=config['lr-warmup-proportion'], total_steps=num_train_optimization_steps)

    # Optimizer
    assert config['optimizer-type'] == 'AdamW'
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = model.collect_params('(?!pool)')

    no_decay = ['bias', 'gamma', 'beta']
    for name, param in param_optimizer.items():
        if any(nd in name for nd in no_decay):
            param.wd_mult = 0

    optimizer = AdamW(lr_scheduler=scheduler, beta1=0.9, beta2=0.999, learning_rate=None,
                      epsilon=1e-6, wd=config['weight-decay'])
    trainer = mxnet.gluon.Trainer(param_optimizer, optimizer=optimizer)
    
    # Record the Total time for train
    total_train_time = 0
    for epoch in range(int(args.num_train_epochs)):
        # Log some infomations
        print(f"--------Epoch: {epoch:03}, " +
              f"lr: {trainer.learning_rate:f}--------")
        
        # Training loop
        train_iter = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar)
        start_time = time.time()
        for batch in train_iter:
            # Move to device
            batch = tuple(t.copyto(mxnet.gpu()) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch
            # Compute prediction and loss
            with mxnet.autograd.record():
                start_logits, end_logits = model(input_ids, segment_ids, input_mask)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.shape[1]
                start_positions = start_positions.clip(0, ignored_index)
                end_positions = end_positions.clip(0, ignored_index)
                # WARNING: NO ignore index in mxnet
                loss_fct = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
            # Backpropagation
            loss.backward()
            # Update (TODO: gradient clipping max_grad_norm=1.0)
            trainer.step(config["train-batch-size"])

        final_loss = loss.asscalar()
        total_train_time += time.time() - start_time
        print(f"Train: Loss(last step): {final_loss:>.4e}, " + 
            f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")

        if not args.is_prof:
            # Validation process
            all_results = []
            eval_iter = tqdm(eval_dataloader, desc="Iteration", disable=args.disable_progress_bar)
            for batch in eval_iter:
                # Move to device
                batch = tuple(t.copyto(mxnet.gpu()) for t in batch)
                input_ids, input_mask, segment_ids, example_indices = batch
                # Forward computing
                with mxnet.autograd.pause():
                    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].asnumpy().tolist()
                    end_logits = batch_end_logits[i].asnumpy().tolist()
                    eval_feature = eval_features[example_index.asscalar()]
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
