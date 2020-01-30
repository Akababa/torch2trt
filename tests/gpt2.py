from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_gpt2 import load_tf_weights_in_gpt2, gelu
from transformers.modeling_utils import PreTrainedModel, prune_conv1d_layer

logger = logging.getLogger(__name__)

DynamicSizes = namedtuple("DynamicSizes", "batch input_ids past")
GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin", }


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self._weightT = None
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x, ds=None):
        if self._weightT is None:
            self._weightT = self.weight.T
        return torch.nn.functional.linear(x, self._weightT, self.bias)


# torch.nn.Conv1d
class Attention(nn.Module):
    def __init__(self, n_embd, n_ctx, config):
        super(Attention, self).__init__()
        # in Attention: n_embd=768 (nx=n_embd)
        # [switch nx => n_embd from Block to Attention to keep identical to TF implem]
        assert n_embd % config.n_head == 0
        # self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer("tmask", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.bool)))
        self.register_buffer("m1e4", torch.full((1, 1, 1), -1e4))
        self.n_head = config.n_head
        self.n_embd = n_embd

        self.c_attn = Conv1D(n_embd * 3, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)
        self._ds = None

    def _attn(self, q, k, v, ds):
        w = torch.matmul(q, k)
        w /= math.sqrt(v.size(-1))
        if self._ds != ds:
            self._ds = ds
            tot_len = ds.input_ids + ds.past
            self._mask = self.tmask[None, ds.past:tot_len, :tot_len]

        w = torch.where(self._mask, w, self.m1e4)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x: torch.Tensor):
        x = x.permute(1, 0, 2).contiguous()
        new_x_shape = x.size()[:-2] + (self.n_embd,)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.n_embd // self.n_head)  # x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        return x.permute(1, 0, 2)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past, ds=None):
        x = self.c_attn(x)
        x = x.view((x.size(0), 3, self.n_embd))
        query, key, value = x[:, 0], x[:, 1], x[:, 2]
        # query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key)  # , k=True)
        value = self.split_heads(value)

        past_value = layer_past[1]  # transpose back cf below
        value = torch.cat((past_value, value), dim=-2)

        past_key = layer_past[0]  # .transpose(-2, -1)
        key = torch.cat((past_key, key), dim=-2)  # this one crashes colab
        present = torch.stack([key, value])  # transpose to have same shapes for stacking

        a = self._attn(query, key.transpose(-2, -1), value, ds)
        a = self.merge_heads(a)
        a = self.c_proj(a)

        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        n_embd = config.n_embd
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = torch.nn.GELU()  # This is a possible difference

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2


class Block(nn.Module):
    def __init__(self, n_ctx, config):
        super(Block, self).__init__()
        n_embd = config.n_embd
        self.ln_1 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(n_embd, n_ctx, config)
        self.ln_2 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * n_embd, config)

    def forward(self, x, layer_past, ds=None):
        a, present = self.attn(self.ln_1(x), layer_past, ds=ds)
        x = x + a
        x += self.mlp(self.ln_2(x))  # residual

        return x, present  # x, present


class GPT2PreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = GPT2Config
    pretrained_model_archive_map = GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super(GPT2PreTrainedModel, self).__init__(*inputs, **kwargs)

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


class GPT2Model(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2Model, self).__init__(config)
        self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([Block(config.n_ctx, config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids: torch.Tensor, past: torch.Tensor):
        if input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        input_len = input_ids.size(0)
        past_length = past.size(-2)
        ds = DynamicSizes(-1, input_len, past_length)

        position_embeds = self.wpe.weight.data[past_length:past_length + input_len]

        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds + position_embeds

        presents = []
        for i in range(self.config.n_layer):
            layer_past = past[i]
            trans_block = self.h[i]
            hidden_states, present = trans_block(hidden_states, layer_past=layer_past, ds=ds)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states, torch.stack(presents)


class GPT2LMHeadModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor, **kwargs):
        hidden_states, pasts = self.transformer(input_ids, **kwargs)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits, pasts
