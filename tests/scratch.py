import torch
import torch2trt
import tensorrt as trt
import numpy as np
import gpt2

# from transformers import GPT2LMHeadModel

torch.manual_seed(0)

past_dummy_seq_length = 1
input_dummy_seq_length = 1
ex_batch_size = 1

config = gpt2.GPT2Config(n_layer=1, n_head=1, n_embd=4, vocab_size=10)
model = gpt2.GPT2LMHeadModel(config)  # .from_pretrained("gpt2")
if torch.cuda.is_available():
    model.to(torch.device("cuda"))

input_names = ["input_ids", "past"]
# [batch, sequence]
input_dummy_shape = (ex_batch_size, input_dummy_seq_length)
# [batch, 2, layers, heads, sequence, embed]
past_dummy_shape = (
    ex_batch_size, 2, config.n_layer, config.n_head, past_dummy_seq_length, config.n_embd // config.n_head)

inputs = [torch.zeros(input_dummy_shape, dtype=torch.int64), torch.rand(past_dummy_shape, dtype=torch.float32)]

opt_profile = [None, None]
opt_profile[0] = np.array([input_dummy_shape] * 3)
opt_profile[0][:, -1] = (1, 1, 1024)
# opt_profile[0][:, 0] = (1, 1, 10)

opt_profile[1] = np.array([past_dummy_shape] * 3)  # [(x, x, x) for x in past_dummy_shape]
opt_profile[1][:, -2] = (1, 256, 1024)
# opt_profile[1][:, 0] = (1, 1, 10)

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
                                output_names=["probs", "past"],
                                optimization_profile=opt_profile,
                                # build_flags=flags,
                                fp16_mode=True,
                                max_workspace_size=8 << 30,
                                use_DLA=False,
                                log_level=trt.Logger.INFO,
                                debug_sync=True)
