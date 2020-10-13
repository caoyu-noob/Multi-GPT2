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

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import checkpoint_sequential

logger = logging.getLogger(__file__)

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm


class ConstantPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, padding_idx):
        super(ConstantPositionalEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.register_buffer('_position_embedding',
                             ConstantPositionalEmbedding.get_embedding(1024,
                                                                       self.embedding_dim))

    @classmethod
    def get_embedding(cls, seq_len, embedding_dim, device=None):
        seq_len += 1

        half_dim = embedding_dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(seq_len, -1)

        if embedding_dim % 2:
            emb = torch.cat([emb, torch.zeros(seq_len, 1)], dim=1)

        return emb

    def forward(self, positions):
        batch_size, seq_len = positions.size()

        cur_seq_len = max(seq_len, torch.max(positions).item())

        if cur_seq_len >= self._position_embedding.size(0):
            self._position_embedding = ConstantPositionalEmbedding.get_embedding(cur_seq_len,
                                                                                 self.embedding_dim,
                                                                                 positions.device)

        return self._position_embedding.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1)


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        nd, ns = size
        max_size = max(nd, ns)
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or any(s<max_size for s in cls._future_mask.shape):
            cls._future_mask = torch.triu(torch.ones(max_size, max_size, dtype=torch.uint8, device=device), 1)

        mask = cls._future_mask[ns-nd:ns, :ns]  # future mask when we already may have past pre-computed values: take a slice at the end of the mask

        return mask

    def __init__(self, n_features, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, apply_future_mask=True, padding_mask=None):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        mask = (w == float('-inf')).all(dim=-1)

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        w.masked_fill_(mask.unsqueeze(-1), 0)

        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value, padding_mask, attn_past_kv=None, attn_past_q=None):
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            apply_future_mask = True  # self-attention
            if attn_past_kv is not None:  # we have already computed part of key, value of this
                past_key, past_value = attn_past_kv[0], attn_past_kv[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
        elif kv_same:
            if attn_past_q is not None:  # we have already computed this query
                query = attn_past_q
            else:
                q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
                query = F.linear(query, q_w, q_b)
            if attn_past_kv is not None:  # we have already computed key, value of this
                key, value = attn_past_kv[0], attn_past_kv[1]
            else:
                kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]
                key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)
            apply_future_mask = False
        else:
            assert False

        save_key_value = (key, value)
        save_query = query

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, apply_future_mask, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x, save_key_value, save_query # we can reuse: key/value for next forward steps, query for next attention ops


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class GatedResidual(nn.Module):
    """ A gated residual layer: see https://arxiv.org/abs/1810.03581
    """
    def __init__(self, in_features):
        super(GatedResidual, self).__init__()
        self.linear_input = nn.Linear(in_features, 1, bias=True)
        self.linear_output = nn.Linear(in_features, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.linear_input.weight, std=0.02)
        nn.init.normal_(self.linear_output.weight, std=0.02)

        self.linear_input.bias.data[:] = 5
        self.linear_output.bias.data[:] = 0

    def forward(self, module_input, module_output):
        gate = torch.sigmoid(self.linear_input(module_input) + self.linear_output(module_output))
        return gate * module_output + (1 - gate) * module_input


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout,
                 successive_attention=False, shared_attention=True, context_size=0):
        super(TransformerBlock, self).__init__()

        if shared_attention:
            context_size = 0

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = LayerNorm(n_features)
        self.context_attns = nn.ModuleList([MultiheadAttention(n_features, n_heads, attn_dropout) for _ in range(context_size)])
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = LayerNorm(n_features)
        self.gated_res = GatedResidual(n_features) if successive_attention else None
        self.dropout = nn.Dropout(dropout)
        self.shared_attention = shared_attention
        self.successive_attention = successive_attention

    def forward(self, x, padding_mask, *contexts, layer_past=None):
        '''contexts = [(context1, padding_mask1), ...]'''

        inputs = (x, padding_mask) + contexts

        result_attns = []
        save_kv = []
        query = None
        if layer_past is None:
            layer_past = [None] * (len(inputs) // 2)

        for i, attn_past_kv in zip(range(0, len(inputs), 2), layer_past):
            c, m = inputs[i], inputs[i+1].byte()

            if self.shared_attention or i == 0:
                attn = self.attn
            else:
                attn = self.context_attns[i // 2 - 1]

            a, key_value, query = attn(x, c, c, m, attn_past_kv=attn_past_kv, attn_past_q=query)
            save_kv.append(key_value)
            result_attns.append(a)

        if self.successive_attention:
            for i, a in enumerate(result_attns):
                a = self.dropout(a)
                x = a + x if i == 0 else self.gated_res(a, x)
        else:
            a = sum(result_attns, 0) / len(result_attns)
            a = self.dropout(a)
            x = x + a

        x = self.attn_norm(x)

        f = self.ff(x)
        f = self.dropout(f)
        x = self.ff_norm(x + f)

        return (x, padding_mask) + contexts + (save_kv,)


class TransformerModule(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size, 
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 normalize_embeddings, n_segments=None, constant_embedding=False,
                 successive_attention=False, sparse_embeddings=False,
                 shared_attention=True, context_size=0):
        super(TransformerModule, self).__init__()

        self._constant_embedding = constant_embedding

        self.embeddings = nn.Embedding(n_embeddings, embeddings_size, padding_idx=padding_idx, sparse=sparse_embeddings)
        if self._constant_embedding:
            self.pos_embeddings = ConstantPositionalEmbedding(embeddings_size, padding_idx=0)
        else:
            self.pos_embeddings = nn.Embedding(n_pos_embeddings, embeddings_size, padding_idx=0,
                                               sparse=sparse_embeddings)
            # self.pos_embeddings = nn.Embedding(n_pos_embeddings + 1, embeddings_size, padding_idx=0, sparse=sparse_embeddings)

        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([TransformerBlock(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout,
                                                      successive_attention, shared_attention, context_size)
                                     for _ in range(n_layers)])
        self.n_segments = n_segments
        self.normalize_embeddings = normalize_embeddings

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        if isinstance(self.pos_embeddings, nn.Embedding):
            nn.init.normal_(self.pos_embeddings.weight, std=0.02)

    def forward(self, x, enc_contexts=[], past=None):
        # x.dim() == 3 if we have additional dialog embeddings else x.dim() == 2
        if past is None: # past store previously computed keys/values for the current generated sentence
            past_length = 0
            past = [None] * len(self.layers)
        else:
            past_length = past[0][0][0].size(-2)  # layer 0, attn ops 0, key (0)

        padding_mask = (x[:, :, 0] if x.dim() == 3 else x).eq(self.embeddings.padding_idx)

        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long) + past_length - 1
        positions.masked_fill_(padding_mask, self.pos_embeddings.padding_idx)

        x = self.embeddings(x)
        if x.dim() == 4: # additional dialog embeddings
            x = x.sum(dim=-2)
        if self.normalize_embeddings:
            x = x * math.sqrt(self.embeddings.embedding_dim)  # Used in pretrained last checkpoint for ConvAI2

        pos_embed = self.pos_embeddings(positions)
        x = x + pos_embed
        x = self.embed_dropout(x)

        enc_contexts = sum(enc_contexts, ())

        if self.n_segments is not None:
            assert False, 'Beam search is not supported'
            padding_mask = padding_mask.float()  # fucking checkpoint_sequential
            padding_mask.requires_grad_()  # fucking checkpoint_sequential
            out = checkpoint_sequential(self.layers, self.n_segments, x, padding_mask, *enc_contexts)
            x = out[0]
        else:
            save_key_values = []
            for layer, layer_past in zip(self.layers, past):
                out = layer(x, padding_mask, *enc_contexts, layer_past=layer_past)
                x = out[0]
                save_key_values.append(out[-1])

        return x, padding_mask, save_key_values
