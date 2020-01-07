import torch
import torch2trt
from transformers import GPT2LMHeadModel, GPT2Config

config = GPT2Config(n_layer=1, n_head=1, n_embd=2, vocab_size=10)
model = GPT2LMHeadModel(config)
input_names = ["input_ids", "past"]
input_shapes = [(1, -1), (2, 1, config.n_head, -1, config.n_embd)]
inputs = []
opt_profile = {}
for name, s in zip(input_names, input_shapes):
    inputs.append(torch.zeros(tuple(x if x != -1 else 1 for x in s)))
    opt_profile[name] = [(x, x, x) for x in s]
opt_profile["input_ids"][1] = (1, 1, 1024)
opt_profile["past"][3] = (1, 256, 1024)

torch2trt.torch2trt(model, inputs=inputs, input_names=input_names, input_shapes=input_shapes, flags=0, fp16_mode=True,
                    optimization_profile=opt_profile)
