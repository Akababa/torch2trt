from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.nn.Embedding.forward')
def convert_Embedding(ctx: ConversionContext):
    module = ctx.method_args[0]  # type: torch.nn.Embedding
    input = ctx.method_args[1]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    weight = module.weight.detach().cpu().numpy()
    # embedding_dim = module.embedding_dim

    layer = ctx.network.add_gather(weight, input_trt, 0)
    layer.num_elementwise_dims = 0
    if not ctx.network.has_implicit_batch_dimension and len(input.shape) == 2:
        layer.num_elementwise_dims = 1

    output._trt = layer.get_output(0)

# @add_module_test(torch.float32, torch.device('cuda'), [(1, 10, 224)])
# def test_Embedding_basic():
#     return torch.nn.Conv1d(10, 5, kernel_size=1, stride=1, padding=0)
