import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask,  \
    get_input_from_batch, get_output_from_batch
from .utils import repeat_along_dim1
from .loss import SoftCrossEntropyLoss

class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=512, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        for i in range(self.num_layers):
            x = self.enc[i](x, mask)

        y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=512, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False, multi_input=False, context_size=1,
                 attention_fusion_type='mean'):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            multi_input: Whether use multiple attention modules in the decoder
            context_size: The number of multiple inputs
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  multi_input,
                  context_size,
                  attention_fusion_type)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.multi_input = multi_input
        self.context_size = context_size

    def forward(self, inputs, encoder_output, mask_src, mask_trg):
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Run decoder
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, hidden_size, vocab, pointer_gen):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab)
        self.p_gen_linear = nn.Linear(hidden_size, 1)
        self.pointer_gen = pointer_gen

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, temp=1, beam_search=False):

        if self.pointer_gen:
            p_gen = self.p_gen_linear(x)
            p_gen = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if self.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = p_gen * vocab_dist

            if isinstance(attn_dist, list):
                enc_batch_extend_vocab_ = [torch.cat([sub_vocab.unsqueeze(1)] * x.size(1),
                                                    1) for sub_vocab in enc_batch_extend_vocab] ## extend for all seq
                if (beam_search):
                    enc_batch_extend_vocab_ = [torch.cat([sub_vocab[0].unsqueeze(0)] * x.size(0),
                                                        0) for sub_vocab in enc_batch_extend_vocab_]  ## extend for all seq
                attn_dist = [F.softmax(a / temp, dim=-1) for a in attn_dist]
                attn_dist_ = [(1 - p_gen) * a / 2 for a in attn_dist]
                for i in range(len(attn_dist_)):
                    vocab_dist_.scatter_add(2, enc_batch_extend_vocab_[i], attn_dist_[i])
                logit = torch.log(vocab_dist_ + 1e-40)
            else:
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)] * x.size(1),
                                                    1)  ## extend for all seq
                if (beam_search):
                    enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0),
                                                        0)  ## extend for all seq
                attn_dist = F.softmax(attn_dist / temp, dim=-1)
                attn_dist_ = (1 - p_gen) * attn_dist
                logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-40)
            return logit
        else:
            return F.log_softmax(logit, dim=-1)

class Embedding:
    def __init__(self, tokenizer, emb_size, pretrained_file, logger):
        self.emb_size = emb_size
        self.embedding = nn.Embedding(tokenizer.n_words, emb_size)
        self.logger = logger
        self.tokenizer = tokenizer
        self.get_pretrained_embedding(pretrained_file)

    def get_pretrained_embedding(self, pretrained_file):
        self.logger.info('Loding embedding from %s', pretrained_file)
        for line in open(pretrained_file, encoding='utf-8').readlines():
            items = line.split()
            if (len(items) == self.emb_size + 1):
                if self.tokenizer.word2idx.__contains__(items[0]):
                    self.embedding.weight.data[self.tokenizer.word2idx[items[0]]] = \
                        torch.tensor([float(x) for x in items[1:]])
        self.embedding.weight.data.requires_grad = True

    def get_embedding(self):
        return self.embedding

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='lstm', num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

    def forward(self, input_emb, hidden):
        output, hidden = self.rnn(input_emb, hidden)
        return output, hidden

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=64, rnn_type='lstm', num_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=num_layers)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_emb, hidden_state, encoder_outputs):
        embedded = self.dropout(input_emb)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_state[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden_state)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

class RNNSeq2Seq(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_layers, tokenizer, pretrained_file, logger, rnn_type='gru'):
        super(RNNSeq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_words
        self.embed_obj = Embedding(tokenizer, emb_dim, pretrained_file, logger)
        self.embedding = self.embed_obj.get_embedding()
        self.encoder = EncoderRNN(emb_dim, hidden_dim, rnn_type=rnn_type, num_layers=num_layers)
        # self.decoder = AttnDecoderRNN(hi)

class TransformerSeq2Seq(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_layers, heads, depth_size, filter_size, tokenizer, pretrained_file,
                 pointer_gen, logger, weight_sharing=True, model_file_path=None, is_eval=False,
                 load_optim=False, label_smoothing=False, multi_input=False, context_size=2,
                 attention_fusion_type='mean'):
        super(TransformerSeq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_words

        self.embed_obj = Embedding(tokenizer, emb_dim, pretrained_file, logger)

        self.embedding = self.embed_obj.get_embedding()
        self.encoder = Encoder(emb_dim, hidden_dim, num_layers=num_layers, num_heads=heads, total_key_depth=depth_size,
                               total_value_depth=depth_size, filter_size=filter_size)

        self.decoder = Decoder(emb_dim, hidden_dim, num_layers=num_layers, num_heads=heads, total_key_depth=depth_size,
                               total_value_depth=depth_size, filter_size=filter_size, multi_input=multi_input,
                               context_size=context_size, attention_fusion_type=attention_fusion_type)
        self.generator = Generator(hidden_dim, self.vocab_size, pointer_gen)
        self.pad_id = tokenizer.pad_id
        self.n_embeddings = tokenizer.n_words
        self.embeddings_size = emb_dim
        self.multi_input = multi_input

        if weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.weight

        self.criterion = nn.NLLLoss(ignore_index=self.pad_id)
        if label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=self.pad_id, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=self.pad_id)
        if is_eval:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()
            self.embedding = self.embedding.eval()

    def forward(self, input_ids, labels, train=True, return_encoded=False):
        label_embeddings = self.embedding(labels)
        mask_target = labels.data.eq(self.pad_id).unsqueeze(1)
        if self.multi_input:
            input_embeddings = [self.embedding(sub_input) for sub_input in input_ids]
            mask_enc = [sub_input.data.eq(self.pad_id).unsqueeze(1) for sub_input in input_ids]
            encoder_outputs = [self.encoder(sub_embeddings, sub_mask) for sub_embeddings, sub_mask in
                               zip(input_embeddings, mask_enc)]
            pre_logits, attn_dist = self.decoder(label_embeddings, encoder_outputs, mask_enc, mask_target)
            logits = self.generator(pre_logits, attn_dist, enc_batch_extend_vocab=input_ids)
        else:
            input_embeddings = self.embedding(input_ids)
            mask_enc = input_ids.data.eq(self.pad_id).unsqueeze(1)
            encoder_outputs = self.encoder(input_embeddings, mask_enc)
            pre_logits, attn_dist = self.decoder(label_embeddings, encoder_outputs, mask_enc, mask_target)
            logits = self.generator(pre_logits, attn_dist, enc_batch_extend_vocab=input_ids)
        if train:
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = self.criterion(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
            if return_encoded:
                return loss, encoder_outputs
            return loss
        else:
            if return_encoded:
                return logits, encoder_outputs
            return logits

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
        beam_scores = beam_scores.unsqueeze(-1) + torch.log(probas + 1e-20) * (1 - is_end.float().unsqueeze(-1))

        if skip_mask is not None:
            beam_scores.masked_fill_(skip_mask, float('-inf'))

        return beam_scores

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

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

    def inference(self, input_ids, encoder_outputs=None):
        if self.inference_mode == 'beam':
            return self.beam_search(input_ids, encoder_outputs)
        elif self.inference_mode == 'sampling':
            return self.sampling_inference(input_ids, encoder_outputs)

    def sampling_inference(self, input_ids, encoder_outputs):
        with torch.no_grad():
            batch_size = input_ids[0].shape[0] if isinstance(input_ids, list) else input_ids.shape[0]
            device = next(self.parameters()).device
            scores = torch.zeros(batch_size, self.response_k, device=device)
            predicts = []
            if self.multi_input:
                mask_enc = [sub_input.data.eq(self.pad_id).unsqueeze(1) for sub_input in input_ids]
                if encoder_outputs is None:
                    input_embeddings = [self.embedding(sub_input) for sub_input in input_ids]
                    encoder_outputs = [self.encoder(sub_embeddings, sub_mask) for sub_embeddings, sub_mask in
                                       zip(input_embeddings, mask_enc)]
            else:
                mask_enc = input_ids.data.eq(self.pad_id).unsqueeze(1)
                if encoder_outputs is None:
                    input_embeddings = self.embedding(input_ids)
                    encoder_outputs = self.encoder(input_embeddings, mask_enc)
            for k in range(self.response_k):
                prevs = torch.full((batch_size, 1), fill_value=self.talker1_id, dtype=torch.long, device=device)
                sample_scores = torch.zeros(batch_size, 1, device=device)
                lens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
                is_end = torch.zeros(batch_size, 1, dtype=torch.uint8, device=device)
                for i in range(self.max_seq_len):
                    label_embeddings = self.embedding(prevs)
                    mask_target = prevs.data.eq(self.pad_id).unsqueeze(1)
                    pre_logits, attn_dist = self.decoder(label_embeddings, encoder_outputs, mask_enc, mask_target)
                    logits = self.generator(pre_logits, attn_dist, enc_batch_extend_vocab=input_ids)[:, -1:, :]
                    probs = self._get_proba_with_temperature(logits.float()).squeeze(1)
                    cur_idxs = torch.multinomial(probs, 1)
                    prevs = torch.cat([prevs, cur_idxs], 1)
                    is_end[cur_idxs == self.eos_id] = 1
                    lens[~is_end] += 1
                    cur_scores = torch.gather(probs, 1, cur_idxs)
                    sample_scores += torch.log(cur_scores)
                sample_scores /= self._length_penalty(lens.float())
                scores[:, k] = sample_scores.squeeze(1)
                cur_predict = []
                for i in range(batch_size):
                    length = lens[i]
                    cur_predict.append(prevs[i, 1: length].tolist())
                predicts.append(cur_predict)
            best_idx = scores.argmax(dim=1)
            final_predicts = []
            for i in range(batch_size):
                final_predicts.append(predicts[best_idx[i]][i])
            return final_predicts

    def beam_search(self, input_ids, encoder_outputs):
        with torch.no_grad():
            batch_size = input_ids[0].shape[0] if isinstance(input_ids, list) else input_ids.shape[0]
            device = next(self.parameters()).device
            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.talker1_id, dtype=torch.long,
                               device=device)
            if self.multi_input:
                mask_enc = [sub_input.data.eq(self.pad_id).unsqueeze(1) for sub_input in input_ids]
                if encoder_outputs is None:
                    input_embeddings = [self.embedding(sub_input) for sub_input in input_ids]
                    encoder_outputs = [self.encoder(sub_embeddings, sub_mask) for sub_embeddings, sub_mask in
                                       zip(input_embeddings, mask_enc)]
                encoder_outputs = [repeat_along_dim1(sub_outputs, self.beam_size) for sub_outputs in encoder_outputs]
                beam_input_ids = [repeat_along_dim1(sub_input_ids, self.beam_size) for sub_input_ids in input_ids]
                beam_mask_enc = [sub_beam_input_ids.data.eq(self.pad_id).unsqueeze(1) for sub_beam_input_ids in
                                 beam_input_ids]
            else:
                mask_enc = input_ids.data.eq(self.pad_id).unsqueeze(1)
                if encoder_outputs is None:
                    input_embeddings = self.embedding(input_ids)
                    encoder_outputs = self.encoder(input_embeddings, mask_enc)
                encoder_outputs = repeat_along_dim1(encoder_outputs, self.beam_size)
                beam_input_ids = repeat_along_dim1(input_ids, self.beam_size)
                beam_mask_enc = beam_input_ids.data.eq(self.pad_id).unsqueeze(1)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)

            for i in range(self.max_seq_len):
                label_embeddings = self.embedding(prevs)
                mask_target = prevs.data.eq(self.pad_id).unsqueeze(1)
                pre_logits, attn_dist = self.decoder(label_embeddings, encoder_outputs, beam_mask_enc, mask_target)
                logits = self.generator(pre_logits, attn_dist, enc_batch_extend_vocab=beam_input_ids)[:, -1:, :]

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

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

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
