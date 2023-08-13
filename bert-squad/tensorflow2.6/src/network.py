# coding=utf-8
# Copyright (c) 2019-2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import re
import copy
import json
import logging
import math
import os
import sys
from io import open

import tensorflow as tf
import numpy as np

from src.config import config

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'ckpt.npz'

def load_numpy_weights_in_bert(model, numpy_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    numpy_path = os.path.abspath(numpy_checkpoint_path)
    print('Loading Numpy file {}'.format(numpy_path))
    weights = np.load(numpy_path)

    for name, array in weights.items():
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        try:
            for m_name in name:
                if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                    l = re.split(r'_(\d+)', m_name)
                else:
                    l = [m_name]
                if l[0] == 'output_bias':
                    pointer = getattr(pointer, 'bias')
                elif l[0] == 'output_weights':
                    pointer = getattr(pointer, 'weight')
                elif l[0] == 'output':
                    pointer = getattr(pointer, 'out')
                else:
                    pointer = getattr(pointer, l[0])
                if len(l) >= 2:
                    num = int(l[1])
                    pointer = pointer[num]
        except AttributeError as e:
            print(f'Warning: could not find the attribute {name} in the model, last {m_name}')
            continue
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'embeddings')
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize TensorFlow weight {}".format(name))
        pointer.assign(tf.convert_to_tensor(array))
    return model

def gelu(x):
    return tf.nn.gelu(x)

def swish(x):
    return x * tf.nn.sigmoid(x)

def layernorm(_):
    return tf.keras.layers.LayerNormalization(epsilon=config['ln-epsilon'])

def linear(_, out_chs, stddev=0.02):
    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev)
    return tf.keras.layers.Dense(out_chs, use_bias=config['dense-use-bias'], kernel_initializer=kernel_initializer)

#torch.nn.functional.gelu(x) # Breaks ONNX export
ACT2FN = {"gelu": gelu, "tanh": tf.nn.tanh,  "relu": tf.nn.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 output_all_encoded_layers=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.output_all_encoded_layers = output_all_encoded_layers
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.type_vocab_size = config.type_vocab_size
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.word_embeddings = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = tf.keras.layers.Embedding(self.max_position_embeddings, self.hidden_size)
        self.token_type_embeddings = tf.keras.layers.Embedding(self.type_vocab_size, self.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = layernorm(self.hidden_size)
        self.dropout = tf.keras.layers.Dropout(self.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids):
        seq_length = tf.shape(input_ids)[1]
        position_ids = tf.range(seq_length, dtype='int64')
        position_ids = tf.expand_dims(position_ids, axis=0)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = linear(config.hidden_size, self.all_head_size)
        self.key = linear(config.hidden_size, self.all_head_size)
        self.value = linear(config.hidden_size, self.all_head_size)

        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] * self.num_attention_heads, self.attention_head_size))
        x = tf.transpose(x, (1, 0, 2))
        return x

    def transpose_key_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(0)
        shape = tf.shape(x)
        x = tf.reshape(x, (shape[0], shape[1] * self.num_attention_heads, self.attention_head_size))
        x = tf.transpose(x, (1, 2, 0))
        return x

    def call(self, hidden_states, attention_mask):
        # (seq, bsz, hidden)
        hshape = tf.shape(hidden_states)
        batch_size = hshape[1]
        seq_length = hshape[0]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer)
        # (bsz, heads, seq, seq)
        attention_scores = tf.reshape(attention_scores, (batch_size,
                                                        self.num_attention_heads,
                                                        seq_length, seq_length))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (bsz, heads, seq, seq)
        attention_probs = self.dropout(attention_probs)
        attention_probs = tf.reshape(attention_probs, (batch_size * self.num_attention_heads,
                                                       seq_length, seq_length))

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, (1, 0, 2))
        # (seq, bsz, hidden)
        context_layer = tf.reshape(context_layer, (seq_length, batch_size, self.all_head_size))

        return context_layer


class BertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = layernorm(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.out = BertSelfOutput(config)

    def call(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.out(self_output, input_tensor)
        return attention_output


class BertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]

    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return self.act(hidden_states)


class BertOutput(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = layernorm(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.out = BertOutput(config)

    def call(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.out(intermediate_output, attention_output)

        return layer_output


class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = [BertLayer(config) for _ in range(config.num_hidden_layers)]
        self.output_all_encoded_layers = config.output_all_encoded_layers

    def call(self, hidden_states, attention_mask):
        all_encoder_layers = []

        # (bsz, seq, hidden) => (seq, bsz, hidden)
        hidden_states = tf.transpose(hidden_states, (1, 0, 2))
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)

            if self.output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        # The hidden states need to be contiguous at this point to enable
        # dense_sequence_output
        # (seq, bsz, hidden) => (bsz, seq, hidden)
        hidden_states = tf.transpose(hidden_states, (1, 0, 2))

        if not self.output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(tf.keras.layers.Layer):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN["tanh"]

    def call(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return self.act(pooled_output)


class BertPreTrainedModel(tf.keras.Model):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (tf.keras.layers.Dense, tf.keras.layers.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.assign(tf.random.normal(mean=0.0, stddev=self.config.initializer_range))
        elif isinstance(module, tf.keras.layers.LayerNormalization):
            module.bias.assign(tf.zeros_like(module.bias))
            module.weight.assign(tf.ones_like(module.weight))
        if isinstance(module, tf.keras.layers.Dense) and module.bias is not None:
            module.bias.assign(tf.zeros_like(module.bias))

    def enable_apex(self, val):
        def _apply_flag(module):
            if hasattr(module, "apex_enabled"):
                module.apex_enabled=val
        self.apply(_apply_flag)

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(
            pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)

        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model, config

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `ckpt.npz` a Numpy dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        logger.info("loading model from path: {}".format(pretrained_model_path))
        # Load config
        config_file = os.path.join(pretrained_model_path, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Build the model
        model(**model.dummy_inputs, training=False)
        # Directly load from a TensorFlow checkpoint
        weights_path = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        return load_numpy_weights_in_bert(model, weights_path)
    
    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        raise NotImplementedError(
            "Model has cross-attention but we couldn't infer the shape for the encoder hidden states. Please manually override dummy_inputs!"
        )


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config)
        self.output_all_encoded_layers = config.output_all_encoded_layers

    def call(self, input_ids, token_type_ids, attention_mask):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask is None:
            attention_mask = tf.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids)

        extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=self.embeddings.word_embeddings.embeddings.dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        
        #see paddle
        #sequence_output = encoded_layers[-1]
        #pooled_output = self.pooler(sequence_output)
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1:]
        return encoded_layers, None#pooled_output


class BertForQuestionAnswering(BertPreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.

    Outputs:
         Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
         position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, max_seq_length=384):
        super(BertForQuestionAnswering, self).__init__(config)
        self.max_seq_length = max_seq_length

        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.qa_outputs = linear(config.hidden_size, 2)  
        # self.apply(self.init_bert_weights)

    def call(self, input_ids, token_type_ids, attention_mask):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = encoded_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, -1)
        end_logits = tf.squeeze(end_logits, -1)
        return start_logits, end_logits
    
    @property
    def dummy_inputs(self):
        dummy = {}
        dummy['input_ids'] = tf.ones(shape=(1, self.max_seq_length), dtype=tf.float32)
        dummy['token_type_ids'] = tf.ones(shape=(1, self.max_seq_length), dtype=tf.float32)
        dummy['attention_mask'] = tf.ones(shape=(1, self.max_seq_length), dtype=tf.float32)
        return dummy