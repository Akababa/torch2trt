import torch
import torch2trt
import tensorrt as trt
from gpt2 import GPT2Config, GPT2LMHeadModel

torch.manual_seed(0)
config = GPT2Config(n_layer=1, n_head=1, n_embd=3, vocab_size=10)
model = GPT2LMHeadModel(config)
input_names = ["input_ids", "past"]

# (batch_size, sequence_length), (batch_size, num_layers, 2, num_heads, sequence_length, embed_size_per_head)
input_shapes = [(1, -1), (1, config.n_layer, 2, config.n_head, -1, config.n_embd)]

inputs = []
inputs.append(torch.zeros((1, 1), dtype=torch.int32))
inputs.append(torch.zeros(tuple(x if x != -1 else 1 for x in input_shapes[1])))

opt_profile = {}
opt_profile["input_ids"] = [(x, x, x) for x in inputs[0].shape]
opt_profile["input_ids"][-1] = (1, 1, 1024)
opt_profile["past"] = [(x, x, x) for x in inputs[1].shape]
opt_profile["past"][-2] = (0, 256, 1024)

with torch.no_grad():
    # probs, pasts = model(input_ids=inputs[0].to(torch.long), past=inputs[1].transpose(0, 1).transpose(1, 2))
    probs, pasts = model(input_ids=inputs[0], past=inputs[1])
    print(probs, pasts)
    # print(inputs[0].shape,inputs[1].shape)
    flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    model_trt = torch2trt.torch2trt(model, inputs=inputs, input_names=input_names, input_shapes=input_shapes,
                                    build_flags=flags,
                                    fp16_mode=True,
                                    optimization_profile=opt_profile)

"""
tensor([[[ 0.0193, -0.0170, -0.0352,  0.0267,  0.0229, -0.0170,  0.0249,
           0.0410,  0.0090,  0.0315]]]) (tensor([[[[[ 0.0000,  0.0000,  0.0000],
           [-0.0006,  0.0084,  0.0323]]]],



        [[[[ 0.0000,  0.0000,  0.0000],
           [-0.0164,  0.0109, -0.0095]]]]]),)
           """