import torch
import torch2trt
import tensorrt as trt
import numpy as np
import gpt2

# import transformers as gpt2

# from transformers import GPT2LMHeadModel

torch.manual_seed(0)

past_dummy_seq_length = 7
input_dummy_seq_length = 11
ex_batch_size = 5

config = gpt2.GPT2Config(n_layer=2, n_head=2, n_embd=4, vocab_size=10)
model = gpt2.GPT2Model(config)
if torch.cuda.is_available():
    model.to(torch.device("cuda"))
# [batch, heads, sequence, embed]
past_dummy_shape = (ex_batch_size, config.n_head, past_dummy_seq_length, config.n_embd // config.n_head)
past_prof = np.array([past_dummy_shape] * 3)  # [(x, x, x) for x in past_dummy_shape]
past_prof[:, -2] = (1, 256, 1024)
# past_prof[:, 0] = (1, 1, 10)

input_names = ["input_ids"]
# [batch, sequence]
# input_shapes = [(-1, -1)]
input_dummy_shape = (ex_batch_size, input_dummy_seq_length)
inputs = [torch.zeros(input_dummy_shape, dtype=torch.int32)]

opt_profile = [None]
opt_profile[0] = np.array([input_dummy_shape] * 3)
opt_profile[0][:, -1] = (1, 1, 1024)
# opt_profile[0][:, 0] = (1, 1, 10)

for kv in "kv":
    for layer_idx in range(config.n_layer):
        input_name = f"past_{layer_idx}_{kv}"
        input_names.append(input_name)
        # input_shapes.append((-1, config.n_head, -1, config.n_embd // config.n_head))
        inputs.append(torch.zeros(past_dummy_shape))
        opt_profile.append(past_prof)

with torch.no_grad():
    # probs, pasts = model(input_ids=inputs[0].to(torch.long),
    #                      past=torch.zeros((config.n_layer, 2, ex_batch_size, config.n_head, past_dummy_seq_length,
    #                                        config.n_embd // config.n_head)))  # inputs[1].transpose(0, 1).transpose(1, 2))
    # probs, pasts = model(**{name: value for name, value in zip(input_names, inputs)})
    # print(probs, pasts)
    # print(inputs[0].shape,inputs[1].shape)
    # flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    model_trt = torch2trt.torch2trt(model,
                                    inputs=inputs,
                                    input_names=input_names,
                                    output_names=["probs"],
                                    optimization_profile=opt_profile,
                                    # build_flags=flags,
                                    fp16_mode=True,
                                    max_workspace_size=24 << 30)
