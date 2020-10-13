# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
"""PyTorch OpenAI GPT model."""


import json
import logging
import math
import os
import random

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from copy import deepcopy

from transformers.activations import gelu_new, swish
from transformers.configuration_openai import OpenAIGPTConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer
from .utils import repeat_along_dim1


logger = logging.getLogger(__name__)

OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin"
}


def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path):
    """ Load tf pre-trained weights in a pytorch model (from NumPy arrays here)
    """
    import re
    import numpy as np

    if ".ckpt" in openai_checkpoint_folder_path:
        openai_checkpoint_folder_path = os.path.dirname(openai_checkpoint_folder_path)

    logger.info("Loading weights from {}".format(openai_checkpoint_folder_path))

    with open(openai_checkpoint_folder_path + "/parameters_names.json", "r", encoding="utf-8") as names_handle:
        names = json.load(names_handle)
    with open(openai_checkpoint_folder_path + "/params_shapes.json", "r", encoding="utf-8") as shapes_handle:
        shapes = json.load(shapes_handle)
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(openai_checkpoint_folder_path + "/params_{}.npy".format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]

    # This was used when we had a single embedding matrix for positions and tokens
    # init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
    # del init_params[1]
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.tokens_embed.weight.shape == init_params[1].shape
        assert model.positions_embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.tokens_embed.weight.shape, init_params[1].shape)
        e.args += (model.positions_embed.weight.shape, init_params[0].shape)
        raise

    model.tokens_embed.weight.data = torch.from_numpy(init_params[1])
    model.positions_embed.weight.data = torch.from_numpy(init_params[0])
    names.pop(0)
    # Pop position and token embedding arrays
    init_params.pop(0)
    init_params.pop(0)

    for name, array in zip(names, init_params):  # names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "w":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


ACT_FNS = {"relu": nn.ReLU, "swish": swish, "gelu": gelu_new}


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, fuse_attention=False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.output_attentions = config.output_attentions

        if not fuse_attention:
            self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, future_mask=True):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        if future_mask:
            b = self.bias[:, :, w.size(-1) - w.size(-2): w.size(-1), : w.size(-1)]
            w = w * b + -1e5 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask.unsqueeze(1).unsqueeze(2)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, k=None, layer_past=None, prev_query=None, attention_mask=None, head_mask=None):
        if k is None:
            x = self.c_attn(x)
            query, key, value = x.split(self.split_size, dim=2)
            query = self.split_heads(query)
            key = self.split_heads(key, k=True)
            value = self.split_heads(value)
            if layer_past is not None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
        else:
            proj_weight, proj_bias = self.c_attn.weight, self.c_attn.bias
            if prev_query is not None:
                query = prev_query
            else:
                size_out = x.size()[:-1] + (self.split_size,)
                query = torch.addmm(proj_bias[: self.split_size], x.view(-1, x.size(-1)),
                                    proj_weight[:, :self.split_size])
                query = query.view(*size_out)
                query = self.split_heads(query)
            if layer_past is None:
                enc_context = k[0]
                size_out = enc_context.size()[:-1] + (self.split_size * 2,)
                key = torch.addmm(proj_bias[self.split_size:], enc_context.view(-1, enc_context.size(-1)), proj_weight[:, self.split_size:])
                key = key.view(*size_out)
                key, value = key.split(self.split_size, dim=2)
                key = self.split_heads(key, k=True)
                value = self.split_heads(value)
            else:
                key, value = layer_past[0].transpose(-2, -1), layer_past[1]
            padding_mask = k[1]
            attention_mask = padding_mask.float() * float('-1e5')
        saved_query = query
        presents = torch.stack((key.transpose(-2, -1), value))

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, future_mask=(k is None))
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, saved_query, presents] + attn_outputs[1:]
        return outputs  # a, (attentions)

    def fuse_qkv(self, query, key, value):
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        w = torch.sqrt(torch.matmul(torch.abs(query), torch.abs(key)))
        w = torch.sign(torch.matmul(torch.abs(query), torch.abs(key))) * w / math.sqrt(value.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd: ns, :ns]
        w = w * b - 1e4 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        a = torch.matmul(w, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[config.afn]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.nx = nx
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.shared_attention = config.shared_attention
        self.dropout = nn.Dropout(config.attn_pdrop)
        self.context_size = config.context_size
        if self.context_size > 0 and not self.shared_attention:
            self.context_attns = nn.ModuleList([Attention(nx, n_ctx, config, scale) for _ in range(self.context_size)])
        self.attention_module = Attention(nx, n_ctx, config, scale, fuse_attention=True)
        self.attention_fusion_type = 'mean'
        # self.attention_weight = nn.Parameter(torch.ones(3, 1) / 3)

    def attention_pooling(self, attention_list):
        if self.attention_fusion_type == "mean":
            return torch.mean(torch.stack(attention_list), dim=0)
        elif self.attention_fusion_type == "max":
            return torch.max(torch.stack(attention_list), dim=0)[0]
        elif self.attention_fusion_type == "min":
            return torch.min(torch.stack(attention_list), dim=0)[0]
        elif self.attention_fusion_type == "sw":
            return torch.mean(torch.stack(attention_list) * self.attention_module.unsqueeze(-1).unsqueeze(-1), dim=0)
        elif self.attention_fusion_type == 'dw':
            return torch.mean(torch.stack(attention_list) * self.attention_module.unsqueeze(1).unsqueeze(1), dim=0)
        elif self.attention_fusion_type == 'linear':
            return self.attention_module(torch.cat(attention_list, dim=-1))
        elif self.attention_fusion_type == 'att':
            return self.attention_module.fuse_qkv(attention_list[1], attention_list[2], attention_list[0])

    def get_attention_pooling_module(self):
        if self.attention_fusion_type == 'sw':
            self.attention_module = torch.nn.Parameter(torch.ones(3, 1) / 3)
        elif self.attention_fusion_type == 'dw':
            self.attention_module = torch.nn.Parameter(torch.ones(3, self.nx) / 3)
        elif self.attention_fusion_type == 'linear':
            self.attention_module = nn.Linear(self.nx * 3, self.nx)
            weight = torch.cat([torch.eye(self.nx) for i in range(3)], dim=0)
            self.attention_module.weight = nn.Parameter(weight.transpose(1, 0))
            self.attention_module.bias = nn.Parameter(torch.zeros(self.nx))
        elif self.attention_fusion_type == 'att':
            self.attention_module.c_proj.weight = nn.Parameter(torch.ones(self.nx, self.nx) / self.nx)
            self.attention_module.c_proj.bias = nn.Parameter(torch.zeros(self.nx))
        else:
            del self.attention_module

    def forward(self, x, enc_contexts=[], layer_past=None, attention_mask=None, head_mask=None):
        x_layer_past = layer_past
        if isinstance(layer_past, list):
            x_layer_past = layer_past[0]
        attn_outputs = self.attn(x, layer_past=x_layer_past, attention_mask=attention_mask, head_mask=head_mask)
        a = attn_outputs[0]
        presents = attn_outputs[2:]

        if len(enc_contexts) != 0:
            context_attention = []
            prev_query = attn_outputs[1]
            for i, enc in enumerate(enc_contexts):
                cur_layer_past = None
                if layer_past is not None:
                    cur_layer_past = layer_past[i + 1]
                if self.shared_attention:
                    enc_output_attn = self.attn(x, k=enc_contexts[i], layer_past=cur_layer_past, prev_query=prev_query)
                else:
                    enc_output_attn = self.context_attns[i](x, k=enc_contexts[i], layer_past=cur_layer_past,
                                                            prev_query=prev_query)
                context_attention.append(enc_output_attn[0])
                presents.append(enc_output_attn[2])
            # if hasattr(self, 'attention_weight'):
            #     a = torch.mean(torch.stack([a] + context_attention) * self.attention_weight.unsqueeze(-1).unsqueeze(-1), dim=0)
            # else:
            a = self.attention_pooling([a] + context_attention)
            # a = torch.mean(torch.stack([a] + context_attention), dim=0)
            a = self.dropout(a)

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = [h, presents]
        return outputs


class OpenAIGPTPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = OpenAIGPTConfig
    pretrained_model_archive_map = OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_openai_gpt
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


OPENAI_GPT_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.OpenAIGPTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

OPENAI_GPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.OpenAIGPTTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        input_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.",
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

        self.init_weights()

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_callable(OPENAI_GPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        enc_contexts=[],
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.OpenAIGPTConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
        import torch

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = OpenAIGPTModel.from_pretrained('openai-gpt')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # new_enc_contexts = []
        # if len(enc_contexts) != 0:
        #     for i, enc in enumerate(enc_contexts):
        #         if isinstance(enc, tuple):
        #             enc = enc[0]
        #         new_enc_contexts.append(enc.view(-1, enc.size()[-2], enc.size()[-1]))

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        if position_ids is None:
            # Code is different from when we had a single embedding matrice from position and token embeddings
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, past_length + input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])


        # Attention mask.
        # if attention_mask is not None:
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #
        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and -10000.0 for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = ()
        all_hidden_states = ()
        presents = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, enc_contexts=enc_contexts, layer_past=layer_past,
                            attention_mask=attention_mask, head_mask=head_mask[i])
            hidden_states = outputs[0]
            present = outputs[1]
            presents = presents + (present, )
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        outputs = (hidden_states.view(*output_shape),)
        outputs = outputs + (presents, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (all hidden states), (all attentions)


@add_start_docstrings(
    """OpenAI GPT Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings). """,
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings_to_callable(OPENAI_GPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.OpenAIGPTConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
        import torch

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, (all hidden states), (all attentions)


@add_start_docstrings(
    """OpenAI GPT Model transformer with a language modeling and a multiple-choice classification
    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.
    The language modeling head has its weights tied to the input embeddings,
    the classification head takes as input the input of a specified classification token index in the input sequence).
""",
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTEncoderDecoderModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = OpenAIGPTModel(config)
        self.encoder = self.transformer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.shared_module = config.shared_module
        self.shared_attention = config.shared_attention
        self.context_size = config.context_size
        if self.shared_module:
            print('Shared')
        else:
            print('Not shared')

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def reload_module_dict(self):
        if not self.shared_module:
            self.encoder = deepcopy(self.transformer)
            for block in self.encoder.h:
                if hasattr(block, 'context_attns'):
                    del block.context_attns
        else:
            if hasattr(self, 'encoder'):
                del self.encoder
        for block in self.transformer.h:
            if not self.shared_attention:
                for context_attn in block.context_attns:
                    context_attn.load_state_dict(block.attn.state_dict())
            else:
                if hasattr(block, 'context_attns'):
                    del block.context_attns
                block.shared_attention = True
            block.attention_fusion_type = self.attention_fusion_type
            block.get_attention_pooling_module()
            # if not self.attention_weight and hasattr(block, 'attention_weight'):
            #     del block.attention_weight

    @add_start_docstrings_to_callable(OPENAI_GPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        persona_ids=None,
        history_ids=None,
        past=None,
        persona_past=None,
        history_past=None,
        attention_mask=None,
        token_type_ids=None,
        persona_token_type_ids=None,
        history_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        r"""
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`, defaults to :obj:`None`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.OpenAIGPTConfig`) and inputs:
        lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``lm_labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`multiple_choice_labels` is provided):
            Multiple choice classification loss.
        lm_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import OpenAIGPTTokenizer, OpenAIGPTDoubleHeadsModel
        import torch

        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
        tokenizer.add_special_tokens({'cls_token': '[CLS]'})  # Add a [CLS] to the vocabulary (we should train it also!)
        model.resize_token_embeddings(len(tokenizer))

        choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        mc_token_ids = torch.tensor([input_ids.size(-1)-1, input_ids.size(-1)-1]).unsqueeze(0)  # Batch size 1

        outputs = model(input_ids, mc_token_ids=mc_token_ids)
        lm_prediction_scores, mc_prediction_scores = outputs[:2]

    """
        contexts = []
        if persona_ids is not None:
            enc_persona_output = self.encoder(persona_ids, token_type_ids=persona_token_type_ids)
            contexts.append(enc_persona_output[0])
            if lm_labels is not None:
                loss_fct = CrossEntropyLoss()
                lm_persona_logits = self.lm_head(enc_persona_output[0])
                shift_persona_logits = lm_persona_logits[..., :-1, :].contiguous()
                shift_persona_labels = persona_ids[..., 1:].contiguous()
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, (all hidden_states), (attentions)

    def encode(self, x):
        input_ids = x[:, :, 0].squeeze(-1)
        token_type_ids = x[:, :, 1].squeeze(-1)
        attention_mask = torch.zeros_like(input_ids).float().masked_fill_(input_ids.eq(self.padding_idx), float('-1e5'))
        x, _ = self.transformer(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) if \
            self.shared_module else self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        padding_mask = input_ids.eq(self.padding_idx)
        return x, padding_mask

    def decode(self, x, enc_contexts=[]):
        input_ids = x[:, :, 0].squeeze(-1)
        token_type_ids = x[:, :, 1].squeeze(-1)
        x, *_ = self.transformer(input_ids, token_type_ids=token_type_ids, enc_contexts=enc_contexts)
        padding_mask = input_ids.eq(self.padding_idx)
        return self.generate(x), x, padding_mask

    def generate(self, enc_x):
        return self.lm_head(enc_x)

    def classify(self, x, padding_mask):
        cls_index = padding_mask.size(-1) - 1 - torch.sum(padding_mask, -1)
        return self.multiple_choice_head(x, cls_index)

    def decode_classify(self, x, enc_contexts=[]):
        input_ids = x[:, :, 0].squeeze(-1)
        token_type_ids = x[:, :, 1].squeeze(-1)
        x, _ = self.transformer(input_ids, token_type_ids=token_type_ids, enc_contexts=enc_contexts)
        cls_index = input_ids.size(-1) - 1 - torch.sum(input_ids.eq(self.padding_idx), -1)
        return self.multiple_choice_head(x, cls_index)

    def _get_proba_with_temperature(self, logits):
        if self.bs_temperature != 1:
            logits /= self.bs_temperature

        return torch.nn.functional.softmax(logits, dim=-1)

    def _get_beam_scores(self, probas, beam_scores, is_end):
        skip_mask = None

        if self.bs_nucleus_p > 0:
            assert self.annealing_topk is None

            sorted_probas, idxs = torch.sort(probas, descending=True, dim=-1)
            skip_mask = torch.cumsum(sorted_probas.cumsum(dim=-1) > self.bs_nucleus_p, dim=-1) > 1
            sorted_probas.masked_fill_(skip_mask, 0.0)
            _, idxs = torch.sort(idxs, dim=-1)
            probas = torch.gather(sorted_probas, -1, idxs)
            skip_mask = torch.gather(skip_mask, -1, idxs)

        beam_scores = beam_scores.unsqueeze(-1) + torch.log(probas) * (1 - is_end.float().unsqueeze(-1))

        if skip_mask is not None:
            beam_scores.masked_fill_(skip_mask, float('-1e5'))

        return beam_scores

    def _sample(self, beam_scores, num_samples, sample_prob=1.):
        if random.random() < sample_prob:
            beam_probas = torch.nn.functional.softmax(beam_scores, dim=-1)
            if self.annealing_topk is not None:
                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                idxs = torch.multinomial(beam_probas, num_samples)
                idxs = torch.gather(sample_idxs, 1, idxs)
            else:
                idxs = torch.multinomial(beam_probas, num_samples)
            scores = torch.gather(beam_scores, 1, idxs)
        else:
            scores, idxs = beam_scores.topk(num_samples, dim=-1)

        return scores, idxs

    def _fix_past(self, past, beam_idxs):
        for layer_output in past:
            for context in layer_output:
                for v in context:
                    size_ = v.size()
                    tile_size = size_[-2] * size_[-1] * size_[-3]
                    new_v = v.contiguous().view(-1, self.beam_size, tile_size)
                    new_v = new_v.gather(1, beam_idxs.unsqueeze(-1).repeat([1, 1, tile_size]))
                    v[...] = new_v.view(*size_)
        return past

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def beam_search(self, enc_contexts=[], return_beams=False, beam_starts=None):
        with torch.no_grad():
            if len(enc_contexts) == 0 and beam_starts is None:
                return []

            batch_size = enc_contexts[0][0].shape[0] if beam_starts is None else beam_starts.shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

            if beam_starts is not None:
                beam_starts = repeat_along_dim1(beam_starts, self.beam_size)
            beam_enc_contexts = repeat_along_dim1(enc_contexts, self.beam_size)
            # beam_enc_contexts = [e[0] for e in beam_enc_contexts]

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)
            past = None

            max_seq_len = min(
                self.n_pos_embeddings - prevs.shape[1] - (beam_starts.shape[1] if beam_starts is not None else 0),
                self.max_seq_len)

            for i in range(max_seq_len):
                inputs = prevs[:, -1:, ...]  # only use the last token (rest is in past)
                token_type_ids = torch.full_like(inputs, self.sent_dialog_id)
                # if self.dialog_embeddings and inputs.dim() < 3:
                #     inputs = torch.stack((inputs, torch.full_like(inputs, self.sent_dialog_id)), dim=inputs.dim())
                # if i == 0 and beam_starts is not None:
                #     inputs = torch.cat((beam_starts, inputs), dim=1)

                outputs, past = self.transformer(inputs, token_type_ids=token_type_ids, enc_contexts=beam_enc_contexts, past=past)

                logits = self.generate(outputs[:, -1, :])

                probs = self._get_proba_with_temperature(logits.float())
                probs = probs.view(batch_size, self.beam_size, -1)

                beam_scores = self._get_beam_scores(probs, beam_scores, is_end)
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float()).unsqueeze(-1)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        g_scores, g_idxs = self._sample(g_beam_scores, group_size, sample_prob=current_sample_prob)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1,
                                                       torch.fmod(g_idxs, self.n_embeddings),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                sym_idxs = torch.fmod(idxs, probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                if self.vocab is not None:
                    logger.info(
                        '\nbeams:\n' + '\n'.join(self.vocab.ids2string(t.detach().cpu().tolist()) for t in prevs))
                    logger.info('\ntop-options:\n' + '\n'.join(self.vocab.ids2string(t.detach().cpu().tolist())
                                                               + str(bi.detach().cpu().tolist()) for t, bi in
                                                               zip(sym_idxs, beam_idxs)))

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                past = self._fix_past(past, beam_idxs)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                return result, beam_lens

            if self.sample:
                probs = torch.nn.functional.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())

        return predicts
