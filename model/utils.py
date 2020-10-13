#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
import io
import json
import os
import random
import re
import sys
import logging
from collections import Counter, namedtuple

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import RectBivariateSpline
from torch.utils.checkpoint import checkpoint

py_version = sys.version.split('.')[0]
if py_version == '2':
    open = io.open
    unicode = unicode
else:
    unicode = str
    open = open


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def repeat_along_dim1(obj, repetitions):
    """ repeat (a possibly nested object of) tensors from (batch, ...) to (batch * repetitions, ...) """
    if isinstance(obj, tuple):
        return tuple(repeat_along_dim1(o, repetitions) for o in obj)
    if isinstance(obj, list):
        return list(repeat_along_dim1(o, repetitions) for o in obj)

    obj = obj.unsqueeze(1).repeat([1, repetitions] + [1] * len(obj.size()[1:]))
    return obj.view(-1, *obj.size()[2:])


def pad_sequence(sequences, batch_first=False, padding_value=0, left=False):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    if not len(sequences):
        return torch.empty(0)
    trailing_dims = sequences[0].size()[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        s_slice = slice(-length, None) if left else slice(None, length)
        s_slice = (i, s_slice) if batch_first else (s_slice, i)
        out_tensor[s_slice] = tensor

    return out_tensor


def checkpoint_sequential(functions, segments, *inputs):
    def run_function(start, end, functions):
        def forward(*inputs):
            for j in range(start, end + 1):
                inputs = functions[j](*inputs)
            return inputs
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(functions) - 1, functions)(*inputs)


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]

    if average:
        return sum(scores) / len(scores)

    return scores


def openai_transformer_config():
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    cfg = dotdict({'n_layers': 12, 'n_embeddings': 40477, 'n_pos_embeddings': 512, 
                   'embeddings_size': 768, 'n_heads': 12, 'dropout': 0.1,
                   'embed_dropout': 0.1, 'attn_dropout': 0.1, 'ff_dropout': 0.1})

    return cfg


def load_openai_weights(model, directory, n_special_tokens=0, use_tokenizer=False):
    # TODO: add check of shapes

    parameters_names_path = os.path.join(directory, 'parameters_names.json')
    parameters_shapes_path = os.path.join(directory, 'parameters_shapes.json')
    parameters_weights_paths = [os.path.join(directory, 'params_{}.npy'.format(n)) for n in range(10)]

    with open(parameters_names_path, 'r') as parameters_names_file:
        parameters_names = json.load(parameters_names_file)

    with open(parameters_shapes_path, 'r') as parameters_shapes_file:
        parameters_shapes = json.load(parameters_shapes_file)

    parameters_weights = [np.load(path) for path in parameters_weights_paths]
    parameters_offsets = np.cumsum([np.prod(shape) for shape in parameters_shapes])
    parameters_weights = np.split(np.concatenate(parameters_weights, 0), parameters_offsets)[:-1]
    parameters_weights = [p.reshape(s) for p, s in zip(parameters_weights, parameters_shapes)]

    if not use_tokenizer:
        parameters_weights[1] = parameters_weights[1][1:] # skip 0 - <unk>

    if isinstance(model.pos_embeddings, nn.Embedding):
        if model.pos_embeddings.num_embeddings - 1 > parameters_weights[0].shape[0]:
            xx = np.linspace(0, parameters_weights[0].shape[0], model.pos_embeddings.num_embeddings - 1)
            new_kernel = RectBivariateSpline(np.arange(parameters_weights[0].shape[0]),
                                             np.arange(parameters_weights[0].shape[1]),
                                             parameters_weights[0])
            parameters_weights[0] = new_kernel(xx, np.arange(parameters_weights[0].shape[1]))

        # parameters_weights[0] = parameters_weights[0][:model.pos_embeddings.num_embeddings - 1]
        # model.pos_embeddings.weight.data[1:] = torch.from_numpy(parameters_weights[0])
        model.pos_embeddings.weight.data = torch.from_numpy(parameters_weights[0])


    if use_tokenizer:
        model.embeddings.weight.data[-n_special_tokens + 1:] = 0
        model.embeddings.weight.data[: -n_special_tokens + 1] = torch.from_numpy(parameters_weights[1])
    else:
        parameters_weights[1] = parameters_weights[1][:model.embeddings.num_embeddings - n_special_tokens]
        model.embeddings.weight.data[:n_special_tokens] = 0
        model.embeddings.weight.data[n_special_tokens:] = torch.from_numpy(parameters_weights[1])

    parameters_weights = parameters_weights[2:]

    for name, weights in zip(parameters_names, parameters_weights):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ':0'
        name = name[:-2]
        name = name.split('/')

        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]

            pointer = getattr(pointer, l[0])

            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]

        if len(weights.shape) == 3: # conv1d to linear
            weights = weights[0].transpose((1, 0))

        pointer.data[...] = torch.from_numpy(weights)

        # Initialize shared attention layer is necessary
        for layer in model.layers:
            attn_state = layer.attn.state_dict()
            for context_attn in layer.context_attns:
                context_attn.load_state_dict(copy.deepcopy(attn_state), strict=False)

def config_logger(log_path):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.INFO)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger
