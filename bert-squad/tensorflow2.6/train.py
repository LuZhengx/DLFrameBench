import os
import json
import sys
import time
import argparse
import ctypes
import subprocess

import tensorflow as tf
from tensorflow.data import Dataset
from tqdm import *

sys.path.append('..')
from common.squad import (load_squad_features, RawResult, get_answers)
from src.config import config
from src.network import BertForQuestionAnswering, BertConfig
from src.optimizer import create_optimizer
# from src.lr_scheduler import LinearWarmUpScheduler


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
    print(f"Load model from pretrained: {args.bert_model}")
    for param in list(model.get_weights()):
        mean = tf.math.reduce_mean(param)
        var = tf.math.reduce_variance(param)
        print(f": {mean:.6f}, {var:.6f}")

    # Train DataLoader
    _, train_features = load_squad_features(args, config["train-file"], True)
    all_input_ids = tf.convert_to_tensor([f.input_ids for f in train_features], dtype='int64')
    all_input_mask = tf.convert_to_tensor([f.input_mask for f in train_features], dtype='int64')
    all_segment_ids = tf.convert_to_tensor([f.segment_ids for f in train_features], dtype='int64')
    all_start_positions = tf.convert_to_tensor([f.start_position for f in train_features], dtype='int64')
    all_end_positions = tf.convert_to_tensor([f.end_position for f in train_features], dtype='int64')
    train_data = Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids,
                                             all_start_positions, all_end_positions))
    # Shuffles records & repeating for training
    train_data = train_data.shuffle(100).batch(
        config["train-batch-size"], drop_remainder=False).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    step_size = int(len(train_features) / config["train-batch-size"])
    num_train_optimization_steps = step_size * args.num_train_epochs
    
    # Eval DataLoader
    eval_examples, eval_features = load_squad_features(args, config["predict-file"], False)
    all_input_ids = tf.convert_to_tensor([f.input_ids for f in eval_features], dtype='int64')
    all_input_mask = tf.convert_to_tensor([f.input_mask for f in eval_features], dtype='int64')
    all_segment_ids = tf.convert_to_tensor([f.segment_ids for f in eval_features], dtype='int64')
    all_example_index = tf.range(all_input_ids.shape[0], dtype='int64')
    eval_data = Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids, all_example_index))
    # Shuffles records & repeating for training
    eval_data = eval_data.batch(config["train-batch-size"], drop_remainder=False).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Optimizer
    assert config['optimizer-type'] == 'AdamW'
    training_vars = model.trainable_variables
    lr = config['lr-base'] * config["train-batch-size"] / config['lr-batch-denom']
    optimizer = create_optimizer(lr, num_train_optimization_steps,
                                 int(num_train_optimization_steps*config['lr-warmup-proportion']), 'adam',     
                                 weight_decay_rate=0.01,
                                 beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-6,
                                 exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    
    @tf.function
    def train_step(batch):
        input_ids, input_mask, segment_ids, start_positions, end_positions = batch
        with tf.GradientTape() as tape:
            # Compute prediction and loss
            start_logits, end_logits = model(input_ids, segment_ids, input_mask, training=True)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = tf.cast(tf.shape(start_logits)[1], 'int64')
            start_positions = tf.clip_by_value(start_positions, 0, ignored_index)
            end_positions = tf.clip_by_value(end_positions, 0, ignored_index)
            start_loss = tf.keras.backend.sparse_categorical_crossentropy(
                start_positions, start_logits, from_logits=True)
            end_loss = tf.keras.backend.sparse_categorical_crossentropy(
                end_positions, end_logits, from_logits=True)
            loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
        # Backpropagation
        grads = tape.gradient(loss, training_vars)
        # Update (TODO: gradient clipping max_grad_norm=1.0)
        optimizer.apply_gradients(zip(grads, training_vars))
        return loss

    @tf.function
    def val_step(batch):
        input_ids, input_mask, segment_ids, _ = batch
        # Forward computing
        return model(input_ids, segment_ids, input_mask, training=False)

    # Record the Total time for train
    total_train_time = 0
    for epoch in range(int(args.num_train_epochs)):
        # Log some infomations
        print(f"--------Epoch: {epoch:03}, " +
              f"lr: {optimizer._decayed_lr(tf.float32).numpy():f}--------")
        
        # Training loop
        train_iter = tqdm(train_data, desc="Iteration", disable=args.disable_progress_bar)
        start_time = time.time()
        for batch in train_iter:
            loss = train_step(batch)

        # Training process end
        total_train_time += time.time() - start_time
        print(f"Train: Loss(last step): {loss:>.4e}, " +
              f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")

        if not args.is_prof:
            # Validation process
            all_results = []
            eval_iter = tqdm(eval_data, desc="Iteration", disable=args.disable_progress_bar)
            for batch in eval_iter:
                batch_start_logits, batch_end_logits = val_step(batch)
                _, _, _, example_indices = batch
                for i, example_index in enumerate(example_indices):
                    start_logits = batch_start_logits[i].numpy().tolist()
                    end_logits = batch_end_logits[i].numpy().tolist()
                    eval_feature = eval_features[example_index.numpy()]
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
