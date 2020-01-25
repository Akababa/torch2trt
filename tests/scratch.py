import torch
from torch import nn
import torch2trt
import tensorrt as trt
import numpy as np

BATCH = True
if not BATCH:
    import gpt2
else:
    import gpt2_batch as gpt2

# import transformers.modeling_gpt2 as gpt2
# from transformers import GPT2LMHeadModel

torch.manual_seed(0)

past_dummy_seq_length = 7
input_dummy_seq_length = 1
ex_batch_size = 1

config = gpt2.GPT2Config()  # n_layer=2, n_head=2, n_embd=4, vocab_size=10)
dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TEST = "gpt2"

# Attention module test
if TEST == "attention":
    model = gpt2.Attention(config.n_embd, config.n_ctx, config)
    input_names = ["x", "layer_past"]
    input_dummy_shape = (ex_batch_size, input_dummy_seq_length, config.n_embd)
    past_dummy_shape = (2, ex_batch_size, config.n_head, past_dummy_seq_length, config.n_embd // config.n_head)
    output_names = ["a", "present"]

# Transformer block test
if TEST == "transformer":
    model = gpt2.Block(config.n_ctx, config)
    input_names = ["x", "layer_past"]
    input_dummy_shape = (ex_batch_size, input_dummy_seq_length, config.n_embd)
    past_dummy_shape = (2, ex_batch_size, config.n_head, past_dummy_seq_length, config.n_embd // config.n_head)
    output_names = ["x_out", "present"]

# GPT2 test
if TEST[:4] == "gpt2":
    model = gpt2.GPT2Model(config).from_pretrained("gpt2")
    input_names = ["input_ids", "past"]
    input_dummy_shape = (ex_batch_size, input_dummy_seq_length)
    past_dummy_shape = (
        ex_batch_size, 2, config.n_layer, config.n_head, past_dummy_seq_length, config.n_embd // config.n_head)
    output_names = ["hidden_states", "presents"] if TEST == "gpt2" else ["probs", "pasts"]

opt_profile = [None, None]
opt_profile[0] = np.array([input_dummy_shape] * 3)
opt_profile[1] = np.array([past_dummy_shape] * 3)  # [(x, x, x) for x in past_dummy_shape]

opt_profile[0][:, 1] = (1, 1, 1)
opt_profile[1][:, -2] = (1, 256, 1024)
opt_profiles = []
opt_profiles.append(opt_profile.copy())
# opt_profile[0][:, 1] = (1, 256, 1024)
# opt_profile[1][:, -2] = (0, 0, 0)
# opt_profiles.append(opt_profile.copy())

inputs = [torch.zeros(input_dummy_shape, dtype=torch.long if TEST[:4] == "gpt2" else dtype, device=device),
          torch.rand(past_dummy_shape, dtype=dtype, device=device)]

model.to(device)
model.to(dtype)

if not BATCH:
    inputs = [inp.squeeze(0) for inp in inputs]
    opt_profiles = [[op_[:, 1:] for op_ in op] for op in opt_profiles]

model(*inputs)
model_trt = torch2trt.torch2trt(model,
                                inputs=inputs,
                                input_names=input_names,
                                output_names=output_names,
                                optimization_profiles=opt_profiles,
                                fp16_mode=True,
                                strict_type_constraints=False,
                                max_workspace_size=1 << 30,
                                use_DLA=True,  # doesn't change much
                                log_level=trt.Logger.VERBOSE,
                                debug_sync=False
                                )
