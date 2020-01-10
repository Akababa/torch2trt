import torch
import torch2trt
import tensorrt as trt
from gpt2 import GPT2Config, GPT2LMHeadModel

torch.manual_seed(0)
config = GPT2Config(n_layer=1, n_head=1, n_embd=3, vocab_size=10)
model = GPT2LMHeadModel(config)
past_dummy_shape = (1, config.n_head, 1, config.n_embd)
past_prof = [(x, x, x) for x in past_dummy_shape]
past_prof[-2] = (0, 256, 1024)

input_names = ["input_ids"]
input_shapes = [(1, -1)]
inputs = [torch.zeros((1, 1), dtype=torch.int32)]
opt_profile = {}
opt_profile["input_ids"] = [(x, x, x) for x in inputs[0].shape]
opt_profile["input_ids"][-1] = (1, 1, 1024)
for kv in "kv":
    for layer_idx in range(config.n_layer):
        input_name = f"past_{layer_idx}_{kv}"
        input_names.append(input_name)
        input_shapes.append((1, config.n_head, -1, config.n_embd))
        inputs.append(torch.zeros(past_dummy_shape))
        opt_profile[input_name] = past_prof

# (batch_size, sequence_length), (batch_size, num_layers, 2, num_heads, sequence_length, embed_size_per_head)

with torch.no_grad():
    # probs, pasts = model(input_ids=inputs[0].to(torch.long), past=inputs[1].transpose(0, 1).transpose(1, 2))
    probs, pasts = model(**{name: value for name, value in zip(input_names, inputs)})
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
