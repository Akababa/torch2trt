from ..conversion_context import *
from torch2trt.module_test import add_module_test


def slice_to_trt(dim_size, dim_slice):
    start = 0 if dim_slice.start is None else dim_slice.start
    stop = dim_size if dim_slice.stop is None else dim_slice.stop
    stride = 1 if dim_slice.step is None else dim_slice.step

    if isinstance(dim_size, trt.ITensor):
        if start == 0 and stride == 1:
            size = dim_size
        else:
            raise NotImplementedError("Only full slice (:) on dynamic axis supported")
    else:
        size = (stop - start - 1) // stride + 1

    return start, size, stride


def num_slice_types(slices):
    num_slice = 0
    for s in slices:
        if isinstance(s, slice) or isinstance(s, int):
            num_slice += 1
    return num_slice


# def list_to_itensor(ctx, lst):
#     consts = ctx.network.add_constant([a if isinstance(a, int) else -1 for a in lst]).get_output(0)
#     varias = ctx.network.add_constant([a if isinstance(a, int) else -1 for a in lst]).get_output(0)


# def dynamify_slices(starts, sizes, strides):
#     if not all(isinstance(x, int) for x in strides):
#         pass

# TODO gather for list
# THIS DOES NOT ASSUME IMPLICIT BATCH DIM, UNLIKE EVERY OTHER CONVERTER
@tensorrt_converter('torch.Tensor.__getitem__')
def convert_tensor_getitem(ctx: ConversionContext):
    input = ctx.method_args[0]
    slices = ctx.method_args[1]
    if isinstance(slices, int) or isinstance(slices, slice):
        slices = (slices,)
    output = ctx.method_return

    input_trt = ctx.get_trt_tensor(input)  # may or may not have a batch dim
    is_const = len(input_trt.shape) == input.dim()  # Only constant tensors have the batch axis

    # Step 1 - Replace ellipsis with expanded slices

    num_ellipsis = input.ndim - num_slice_types(slices)

    num_slices = 0
    new_slices = []
    for s in slices:
        if s == Ellipsis:
            while num_ellipsis > 0:
                new_slices.append(slice(None, None, None))
                num_ellipsis -= 1
        elif isinstance(s, slice):
            new_slices.append(s)
            num_slices += 1
        elif isinstance(s, int):
            new_slices.append(s)
        else:
            raise ValueError(f"unsupported type {type(s)} in __getitem__")

    # fill missing slices at end
    while num_slice_types(new_slices) < len(input.shape):
        new_slices.append(slice(None, None, None))
    # print("new_slices:",new_slices)
    # print("input shape:", input.shape)
    # print("input_trt shape:", input_trt.shape)
    assert len(new_slices) == input.dim()

    if not is_const:
        if new_slices[0] != slice(None, None, None):
            raise ValueError(f"can't slice on batch dimension")  # TODO actually I can in explicit mode
        new_slices = new_slices[1:]
        # print("new_slices shrink to:", new_slices)
        assert len(new_slices) == len(input_trt.shape)

    # # Step 2 - Remove batch from slices (TRT from this point)
    # slices = tuple(new_slices[1:]) # remove batch

    # Step 3 - Add slice layer (will currently ignore 'None' slices)

    starts, sizes, strides = [], [], []
    # input_trt_shape = ctx.network.add_shape(input_trt).get_output(0)
    for i, (s, input_size) in enumerate(zip(new_slices, input_trt.shape)):
        if input_size == -1:
            raise NotImplementedError("sorry no dynamic sizes")
            if s == slice(None, None, None):
                # dynamic input size - only support full slices for now
                input_size = ctx.network.add_slice(
                    input_trt_shape, [i], [1], [1]).get_output(0)

        if isinstance(s, slice):
            start, size, stride = slice_to_trt(input_size, s)
            starts.append(start)
            sizes.append(size)
            strides.append(stride)
        elif isinstance(s, int):
            starts.append(s)
            sizes.append(1)
            strides.append(1)
        else:
            raise ValueError("Invalid slice")
    # starts, sizes, strides = dynamify_slices(starts,sizes,strides)

    # print("starts,sizes,strides:", starts, sizes, strides)
    assert len(starts) == len(sizes) == len(strides) == len(new_slices) == len(input_trt.shape)

    output_trt = ctx.network.add_slice(input_trt, starts, sizes, strides).get_output(0)
    # print("shape after step3:", output_trt.shape)
    assert len(output_trt.shape) >= 0, "Shape Error"  # getitem is prone to these

    # Step 4 - Add shuffle layer to insert dimensions for 'None' slices and remove dimensions for 'int' slices

    # num_non_slice = len([s for s in new_slices if not isinstance(s, slice)])
    if tuple(output_trt.shape) != tuple(output.shape):
        layer = ctx.network.add_shuffle(output_trt)
        # print("reshape to:", output.shape)
        layer.reshape_dims = tuple(output.shape)  # don't exclude batch
        output_trt = layer.get_output(0)

    output._trt = output_trt
    assert len(output_trt.shape) >= 0, "Shape Error"


class LambdaModule(torch.nn.Module):
    def __init__(self, fn):
        super(LambdaModule, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 3)])
def test_tensor_getitem_1d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_int():
    return LambdaModule(lambda x: x[:, 0])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided():
    return LambdaModule(lambda x: x[:, ::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_offset():
    return LambdaModule(lambda x: x[:, 1::2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_strided_range():
    return LambdaModule(lambda x: x[:, 1:3:2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim():
    return LambdaModule(lambda x: x[:, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_insert_dim_ellipsis():
    return LambdaModule(lambda x: x[:, None, ...])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_dim():
    return LambdaModule(lambda x: x[:, ..., None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_append_2dim():
    return LambdaModule(lambda x: x[:, ..., None, None])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 5, 4, 3)])
def test_tensor_getitem_2d_weird_combo():
    return LambdaModule(lambda x: x[:, 0:3:4, None, None, 1, ...])
