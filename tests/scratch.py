import torch
from torch import nn
import torch2trt
import tensorrt as trt
import numpy as np
from transformers import GPT2Model as gpt2orig, GPT2Config

BATCH = False
if not BATCH:
    import gpt2
else:
    import gpt2_batch as gpt2

torch.manual_seed(0)

past_dummy_seq_length = 31
input_dummy_seq_length = 11
ex_batch_size = 3

config = GPT2Config.from_pretrained("gpt2-medium")  # n_layer=2, n_head=2, n_embd=4, vocab_size=10)
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
    model = gpt2.GPT2Model(config)
    input_names = ["input_ids", "past"]
    input_dummy_shape = (ex_batch_size, input_dummy_seq_length)
    past_dummy_shape = (config.n_layer, 2, ex_batch_size, config.n_head, past_dummy_seq_length,
                        config.n_embd // config.n_head)
    output_names = ["hidden_states", "presents"] if TEST == "gpt2" else ["probs", "pasts"]

opt_profile = [None, None]
opt_profile[0] = np.array([input_dummy_shape] * 3)
opt_profile[1] = np.array([past_dummy_shape] * 3)  # [(x, x, x) for x in past_dummy_shape]

opt_profile[0][:, -1] = (1, 1, 64)
opt_profile[1][:, -2] = (1, 256, 1024)
opt_profiles = []
opt_profiles.append(opt_profile.copy())
opt_profile[0][:, -1] = (64, 64, 64)
opt_profile[1][:, -2] = (1, 256, 1024)
opt_profiles.append(opt_profile.copy())

inputs = [torch.zeros(input_dummy_shape, dtype=torch.long if TEST[:4] == "gpt2" else dtype, device=device),
          torch.rand(past_dummy_shape, dtype=dtype, device=device)]

model.to(device)
model.to(dtype)

if not BATCH:
    inputs = [inputs[0][0], inputs[1][:, :, 0]]
    opt_profiles = [[op0[:, 1:], np.delete(op1, 2, axis=1)] for op0, op1 in opt_profiles]

# with torch.no_grad():
#     out_torch = model(*inputs)
#     out_trans = gpt2orig.from_pretrained("gpt2")(inputs[0].unsqueeze(0),
#                                                  inputs[1].unsqueeze(2))  # .transpose(0, 1).unsqueeze(2))
#     out_diff0 = (out_torch[0].unsqueeze(0) - out_trans[0]).numpy()
#     out_diff1 = (out_torch[1].unsqueeze(2) - torch.stack(out_trans[1])).numpy()

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
