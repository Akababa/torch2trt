import torch
import tensorrt as trt
import numpy as np
from .conversion_utils import *
from typing import Union, Tuple, Dict, Optional

CONVERTERS = {}


# put constants with batch dim?
# TODO FIX this by comparing dim sizes of input and input._trt!
class ConversionContext(object):
    __default = object()  # dummy default

    def get_trt_one(self, t: Union[torch.Tensor, float, int]) -> trt.ITensor:
        # GET TRT TENSOR (OR CREATE TRT CONSTANT)
        # get tensor w/ _trt
        if isinstance(t, torch.Tensor) and hasattr(t, '_trt'):
            trt_tensor = t._trt
        # elif isinstance(t, trt.ITensor):
        #     trt_tensor = t
        # or... add constant for leaf tensor w/o _trt
        elif isinstance(t, torch.Tensor) and not hasattr(t, '_trt'):
            # add leaf tensor - don't exclude batch when adding constants...?
            t._trt = self._add_const_trt(t)
            trt_tensor = t._trt
        # or... create and add constant for scalar primitive (lost reference) TODO
        elif isinstance(t, float) or isinstance(t, int):
            dtype = (torch.float32 if isinstance(t, float) else torch.int32)
            trt_tensor = self._add_const_trt(torch.tensor(t, dtype=dtype))
        else:
            raise ValueError(f'Bad tensor of type {type(t)}')

        assert len(trt_tensor.shape) >= 0
        return trt_tensor

    def convert_dtype_to(self, tensor: trt.ITensor, dtype: trt.DataType):
        assert isinstance(tensor, trt.ITensor) and isinstance(dtype, trt.DataType)
        if tensor.dtype == dtype:
            return tensor
        layer = self.network.add_identity(tensor)
        layer.set_output_type(0, dtype)
        return layer.get_output(0)

    def broadcast_together(self, *tensors: trt.ITensor):
        # Make same number of dims and dtype
        assert all(isinstance(t, trt.ITensor) for t in tensors)
        if len(set(t.dtype for t in tensors)) > 1:
            types = [trt.int8, trt.int32, trt.float16, trt.float32]
            max_type = types[max(types.index(t.dtype) for t in tensors)]
            print(f"promoting {[t.dtype for t in tensors]} to {max_type}")  # TODO type promotion
            tensors = [self.convert_dtype_to(t, max_type) for t in tensors]

        broadcast_num_dim = max(len(t.shape) for t in tensors)
        new_tensors = [self.make_broadcastable_to(t, broadcast_num_dim) for t in tensors]
        all_dims_set = [set(nt.shape[i] for nt in new_tensors) - {1, -1} for i in range(broadcast_num_dim)]
        assert all(len(dims_set) <= 1 for dims_set in all_dims_set)
        return new_tensors

    def get_trt_dim(self, name="dim", pos=1, default=__default, ndims=None):
        # If ndims is none, we must have no negative dim indices.
        # Gets the dim argument, possibly shifted for implicit batch dim.
        # default is in terms of torch, and will get converted to trt!
        dim = self.get_arg(name, pos, default=default)
        dim = _fix_dim(dim, ndims)
        return dim

    def get_trt_axes(self, trt_dim, ndims):
        # Returns an axes bitmask
        if isinstance(trt_dim, int):
            trt_dim = (trt_dim,)
        trt_dim = _fix_dim(trt_dim, ndims)
        # create axes bitmask for reduce layer
        axes = 0
        for d in trt_dim:
            assert d >= 0
            axes |= (1 << d)  # don't -1 because it's already trt dim

        return axes

    # If len(trt_tensor.shape) < broadcast_num_dim, prepends 1 dims to match number of dims in shape,
    # otherwise returns trt_tensor
    def make_broadcastable_to(self, trt_tensor: trt.ITensor, broadcast_num_dim: int):
        assert isinstance(trt_tensor, trt.ITensor)
        trt_num_dims = len(trt_tensor.shape)
        assert trt_num_dims <= broadcast_num_dim
        if trt_num_dims < broadcast_num_dim:
            # append 1 size dims to front
            diff = broadcast_num_dim - trt_num_dims
            shape = tuple([1] * diff + list(trt_tensor.shape))
            trt_tensor = self.reshape_to(trt_tensor, shape)
        return trt_tensor

    def reshape_to(self, trt_tensor: trt.ITensor, new_shape):
        assert new_shape.count(-1) <= 1
        layer = self.network.add_shuffle(trt_tensor)
        layer.reshape_dims = new_shape
        return layer.get_output(0)

    def get_arg(self, name, pos, default=__default, to_trt=False):
        if name in self.method_kwargs:
            val = self.method_kwargs[name]
        elif len(self.method_args) > pos:
            val = self.method_args[pos]
        elif default is not ConversionContext.__default:
            val = default
        else:
            raise ValueError(f"Missing arg {name} at pos {pos}")
        if to_trt:
            val = self.get_trt_one(val)
        return val

    def get_dim_of_shape(self, trt_shape: trt.ITensor, trt_dim: int) -> trt.ITensor:
        trt_dyn_shape_dim = self.network.add_slice(trt_shape, (trt_dim,), (1,), (0,)).get_output(0)
        return self.reshape_to(trt_dyn_shape_dim, ())

    def add_inputs(self, torch_inputs, input_shapes=None, names=None):
        if names is None:
            names = ['input_%d' % i for i in range(len(torch_inputs))]
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            if not hasattr(torch_input, '_trt'):
                shape = tuple(torch_input.shape if input_shapes is None else input_shapes[i])
                if self.has_implicit_batch():
                    shape = shape[1:]
                torch_dtype = torch_input.dtype
                if torch_dtype == torch.long:
                    print(f"Info: Found input {names[i]} with dtype torch.long. "
                          f"TRT doesn't support int64 so the converted model "
                          "will use input with dtype int32 instead.")
                    torch_dtype = torch.int32
                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=shape,
                    dtype=torch_dtype_to_trt(torch_dtype),
                )
                trt_tensor.location = torch_device_to_trt(torch_input.device)
                torch_input._trt = trt_tensor

    def mark_outputs(self, torch_outputs, names=None):
        if names is None:
            names = ['output_%d' % i for i in range(len(torch_outputs))]
        self.output_names = names

        for i, torch_output in enumerate(torch_outputs):
            trt_tensor = torch_output._trt
            trt_tensor.name = names[i]
            trt_tensor.location = torch_device_to_trt(torch_output.device)
            trt_tensor.dtype = torch_dtype_to_trt(torch_output.dtype)
            print(f"Found output {trt_tensor.name} with shape {trt_tensor.shape}, dtype {trt_tensor.dtype}")
            self.network.mark_output(trt_tensor)

    def has_implicit_batch(self) -> bool:
        return self.network.has_implicit_batch_dimension

    @property
    def nonbatch_dim(self):
        return 1 if self.has_implicit_batch() else 0

    def _add_const_trt(self, tensor: torch.Tensor):
        shape = tuple(tensor.shape)
        array = tensor.detach().cpu().numpy()
        if array.dtype == np.int64:  # TRT doesn't support long
            # print(f"Warning: implicitly converting an array of shape {array.shape} from int64 to int32")
            array = array.astype(np.int32)
        return self.network.add_constant(shape, array).get_output(0)

    def setup_method(self, args, kwargs, outputs, method_str):
        self.method_args = args
        self.method_kwargs = kwargs
        self.method_return = outputs
        self.method_str = method_str
        self._first_input = None

    def cleanup_method(self):
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.method_str = None
        self._first_input = None

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

    def __init__(self, network: trt.INetworkDefinition, converters=CONVERTERS):
        self.network = network
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.method_str = None
        self._first_input = None
        # We keep a dict so we can track ints for dynamic size
        # self._trt = dict()  # type: Dict[int, trt.ITensor]
        self.hooks = [
            ConversionHook(self, method, converter)
            for method, converter in converters.items()
        ]


class ConversionHook(object):
    """Attaches TensorRT converter to PyTorch method call"""

    def __init__(self, ctx, method, converter):
        self.ctx = ctx
        self.method_str = method
        self.converter = converter

    def _set_method(self, method):
        exec('%s = method' % self.method_str)

    def __enter__(self):
        try:
            self.method_impl = eval(self.method_str)
        except AttributeError:
            self.method_impl = None

        if self.method_impl:
            self._set_method(_attach_converter(self.ctx, self.method_impl, self.converter, self.method_str))

    def __exit__(self, type, val, tb):
        if self.method_impl:
            self._set_method(self.method_impl)


def shape_ok(t: torch.Tensor):
    assert isinstance(t, torch.Tensor)
    return all(torch_d == trt_d or trt_d == -1 for torch_d, trt_d in
               zip(t.shape, t._trt.shape))


# Assume there will NEVER be a dynamic dim index
def _fix_dim(dim, ndims: Optional[int]):
    if dim is None:
        return dim
    if dim == "_all":
        assert isinstance(ndims, int)
        return tuple(range(ndims))

    def helper(d):
        if d < 0:
            d = ndims + d
        assert 0 <= d
        assert ndims is None or d < ndims
        return d

    if isinstance(dim, int):
        return helper(dim)
    else:
        assert isinstance(dim, list) or isinstance(dim, tuple)
        return tuple(helper(d) for d in dim)


def _attach_converter(ctx: ConversionContext, method, converter, method_str):
    """Gets a function that executes PyTorch method and TensorRT converter"""
    global DUMMY_CONVERTERS

    def wrapper(*args, **kwargs):
        skip = True

        # check if another (parent) converter has lock
        if not ctx.lock:
            if converter['is_real']:
                ctx.lock = True  # only real converters can acquire lock
            skip = False

        # run original method
        outputs_orig = method(*args, **kwargs)
        outputs = outputs_orig

        if not skip:
            ctx.setup_method(args, kwargs, outputs, method_str)
            if converter["is_real"]:
                def stringer(t):
                    if isinstance(t, torch.Tensor):
                        return f"Tensor({tuple(t.shape)})"
                    elif isinstance(t, list) or isinstance(t, tuple):
                        return f"{type(t).__name__}[{stringer(t[0]) if len(t) > 0 else 'EMPTY'}]"
                    else:
                        return type(t).__name__

                print("Converting {0}({1})".format(
                    method_str,
                    ", ".join(list(map(stringer, args)) +
                              list(f"{k}={stringer(v)}" for k, v in kwargs.items()))
                ))

            converter['converter'](ctx)

            if not (outputs is ctx.method_return):
                print(f"wrapper overwrote output!")
                assert converter["is_real"]
                outputs = ctx.method_return

            if converter['is_real']:
                def _check_shape_recursive(outputs_recurse, pos=()):
                    # Checks for corrupted converter trt tensor outputs, since python api/abi doesn't do so
                    if isinstance(outputs_recurse, int) or isinstance(outputs_recurse, float):
                        expected = outputs_orig
                        for p in pos:
                            expected = expected[p]
                        assert outputs_recurse == expected
                        print(f"Output const {outputs_recurse}, {type(outputs_recurse).__name__}\n"
                              f"matched expected")
                        return
                    if isinstance(outputs_recurse, torch.Tensor):
                        if hasattr(outputs_recurse, "_trt"):
                            try:
                                len(outputs_recurse._trt.shape)  # This will crash python without exception handling
                                assert shape_ok(outputs_recurse)
                                print(f"Output shape     {tuple(outputs_recurse._trt.shape)}, {outputs_recurse._trt.dtype}\n"
                                      f"matched expected {tuple(outputs_recurse.shape)}, {outputs_recurse.dtype}")
                            except Exception as e:
                                print(f"Error: wrong shape on output {pos} of {method_str}:\n"
                                      f"expected:{tuple(outputs_recurse.shape)}\n"
                                      f"actual:  {tuple(outputs_recurse._trt.shape)}")
                                raise e
                        else:
                            print(f"Warning: output {pos} of {method_str} not supported")
                        return
                    for i, output_i in enumerate(outputs_recurse):
                        _check_shape_recursive(output_i, pos + (i,))

                _check_shape_recursive(ctx.method_return)

            print()
            # convert to None so conversion will fail for unsupported layers
            ctx.cleanup_method()
            ctx.lock = False

        return outputs

    return wrapper


def tensorrt_converter(method: str, is_real=True):
    def register_converter(converter):
        if method in CONVERTERS:
            if CONVERTERS[method]["is_real"]:
                raise AttributeError(f"Overwrote {method}")
        CONVERTERS[method] = {'converter': converter, 'is_real': is_real}
        return converter

    return register_converter
