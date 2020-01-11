from ..conversion_context import *


@tensorrt_converter('torch.Tensor.__iter__')
def convert_iter(ctx: ConversionContext):
    input = ctx.method_args[0]
    input_trt = ctx.get_trt_tensor(input)
    output = ctx.method_return

    loop = ctx.network.add_loop()
    it = loop.add_iterator(input_trt)
    loop.add_trip_limit(input.shape[0], trt.TripLimit.COUNT)

    output._trt = it.get_output(0)
