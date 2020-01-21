from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_gpt2 import load_tf_weights_in_gpt2
from transformers.modeling_utils import PreTrainedModel, prune_conv1d_layer

logger = logging.getLogger(__name__)

GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "gpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin",
    "gpt2-medium": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin",
    "gpt2-large": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin",
    "gpt2-xl": "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-pytorch_model.bin",
    "distilgpt2": "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-pytorch_model.bin", }


# TODO use BERT plugins .cu once i get this working..
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


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
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        nx = self.weight.shape[0]
        # assert x.size(-1) == nx
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, nx), self.weight)
        x = x.view(size_out[0], -1, self.nf)
        return x


# torch.nn.Conv1d
class Attention(nn.Module):
    def __init__(self, n_embd, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        self.output_attentions = config.output_attentions

        # in Attention: n_embd=768 (nx=n_embd)
        # [switch nx => n_embd from Block to Attention to keep identical to TF implem]
        assert n_embd % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_embd
        self.scale = scale

        self.c_attn = Conv1D(n_embd * 3, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w /= math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e4 * (1.0 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        return torch.matmul(w, v)

    def merge_heads(self, x: torch.Tensor):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past):
        x = self.c_attn(x)
        x = x.view(*(x.size()[:-1] + (3, self.split_size)))
        query, key, value = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        # query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        past_key = layer_past[0].transpose(-2, -1)
        past_value = layer_past[1]  # transpose back cf below
        key = torch.cat((past_key, key), dim=-1)  # this one crashes colab
        value = torch.cat((past_value, value), dim=-2)

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        a = self._attn(query, key, value)

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        n_embd = config.n_embd
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        n_embd = config.n_embd
        self.ln_1 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
        self.attn = Attention(n_embd, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * n_embd, config)

    def forward(self, x, layer_past=None):
        ln1_x = self.ln_1(x)
        a, present = self.attn(ln1_x, layer_past=layer_past)
        x = x + a
        x += self.mlp(self.ln_2(x))  # residual

        return x, present  # x, present
        # return outputs  # x, present, (attentions)


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
        # self.device = torch.device("cuda")

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids: torch.Tensor, past: torch.Tensor):
        if input_ids is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # input_ids = input_ids.to(self.device) # do this before
        # past = past.to(self.device)
        batch_size, input_len = input_ids.size()
        past = past.permute((2, 1, 0, 3, 4, 5))
        past_length = past.size(-2)

        position_embeds = self.wpe.weight.data[past_length:past_length + input_len].unsqueeze(0)  # put in the batch

        inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)  # dropout

        presents = []
        for i in range(self.config.n_layer):
            layer_past = past[i]
            trans_block = self.h[i]
            hidden_states, present = trans_block(hidden_states, layer_past=layer_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        # hidden_states = hidden_states.view(batch_size, input_len, self.config.n_embd)

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
        transformer_outputs = self.transformer(input_ids, **kwargs)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
