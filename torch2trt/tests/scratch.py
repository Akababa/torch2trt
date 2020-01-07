import torch
# import torch2trt
import tensorrt as trt
import importlib

importlib.reload(torch2trt)
from .gpt2 import GPT2LMHeadModel, GPT2Config

config = GPT2Config(n_layer=1, n_head=1, n_embd=2, vocab_size=10)
model = GPT2LMHeadModel(config)
input_names = ["input_ids", "past"]
input_shapes = [(1, -1), (config.n_layer, 2, 1, config.n_head, -1, config.n_embd)]
inputs = []
inputs.append(torch.zeros((1, 1), dtype=torch.int32))
inputs.append(torch.zeros(tuple(x if x != -1 else 1 for x in input_shapes[1])))
opt_profile = {}
opt_profile["input_ids"] = [(x, x, x) for x in inputs[0].shape]
opt_profile["past"] = [(x, x, x) for x in inputs[1].shape]
opt_profile["input_ids"][-1] = (1, 1, 1024)
opt_profile["past"][-2] = (0, 256, 1024)

# print(model(input_ids=inputs[0].to(torch.long),past=inputs[1]))
# print(inputs[0].shape,inputs[1].shape)
flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
model_trt = torch2trt.torch2trt(model, inputs=inputs, input_names=input_names, input_shapes=input_shapes, flags=flags,
                                fp16_mode=True,
                                optimization_profile=opt_profile)
