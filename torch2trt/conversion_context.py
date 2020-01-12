import torch
import tensorrt as trt
import numpy as np
from .conversion_utils import *

CONVERTERS = {}


def is_implicit_batch_tensor(t: torch.Tensor):
    return len(t.shape) != len(t._trt.shape)


# TODO remove this
def _check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert (dtype == t.dtype)  # , 'Tensor data types must match')
    # assert (dtype is not None)  # , 'Data type could not be inferred from any item in list')
    if dtype is None and len(tensors) == 1:
        dtype = torch.float32 if isinstance(tensors[0], float) else torch.int32
    assert dtype is not None, "No data type!"
    return dtype


# put constants with batch dim?
# TODO FIX this by comparing dim sizes of input and input._trt!
class ConversionContext(object):
    __default = object()  # dummy default

    def __init__(self, network: trt.INetworkDefinition, converters=CONVERTERS):
        self.network = network
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.method_str = None
        self._first_input = None
        self.hooks = [
            ConversionHook(self, method, converter)
            for method, converter in converters.items()
        ]

    def get_trt_tensor(self, *tensors: torch.Tensor):
        """
        Creates missing TensorRT tensors and adds shuffle layers to make tensors broadcastable
        TRT tensors are missing batch dimension (implicit) EXCEPT for CONSTANTS,
         while pytorch tensors have the first batch dim (explicit)
        """
        # broadcast = len(tensors) > 1
        dtype = _check_torch_dtype(*tensors)
        # get broadcast dimension
        broadcast_num_dim = 0  # 0 dim DOES exist!!
        for t in tensors:
            if isinstance(t, torch.Tensor):
                if not hasattr(t, '_trt'):  # It's a constant!!
                    num_dim = len(t.shape)  # don't exclude batch for constants
                else:  # It's a variable (on input path)
                    num_dim = len(t._trt.shape)  # non-leaf tensors must already have _trt, get shape from that
                if num_dim > broadcast_num_dim:
                    broadcast_num_dim = num_dim
        trt_tensors = []
        for i, t in enumerate(tensors):
            # GET TRT TENSOR (OR CREATE TRT CONSTANT)

            # get tensor w/ _trt
            if isinstance(t, torch.Tensor) and hasattr(t, '_trt'):
                trt_tensor = t._trt
                # trt_tensor._trt_const = False

            # or... add constant for leaf tensor w/o _trt
            elif isinstance(t, torch.Tensor) and not hasattr(t, '_trt'):
                # add leaf tensor
                # don't exclude batch when adding constants...?
                t._trt = self._add_const_trt(t)
                trt_tensor = t._trt
                # trt_tensor._trt_const = True

            # or... add constant for scalar primitive
            elif isinstance(t, float) or isinstance(t, int):
                shape = (1,) * broadcast_num_dim
                scalar = torch.full(shape, t, dtype=dtype)
                trt_tensor = self._add_const_trt(scalar)
                # trt_tensor._trt_const = True

            # assert (trt_tensor is not None)
            assert len(trt_tensor.shape) >= 0

            # MAKE TRT TENSOR BROADCASTABLE IF IT IS NOT ALREADY

            if len(trt_tensor.shape) < broadcast_num_dim:
                # append 1 size dims to front
                diff = broadcast_num_dim - len(trt_tensor.shape)
                shape = tuple([1] * diff + list(trt_tensor.shape))
                layer = self.network.add_shuffle(trt_tensor)
                layer.reshape_dims = shape
                trt_tensor = layer.get_output(0)
                assert len(trt_tensor.shape) >= 0

            trt_tensors.append(trt_tensor)

        return trt_tensors[0] if len(trt_tensors) == 1 else tuple(trt_tensors)

    def get_trt_dim(self, name="dim", pos=1, default=__default):
        # Gets the dim argument, possibly shifted for implicit batch dim.
        # default is in terms of torch, and will get converted to trt!
        torch_dim = self.get_arg(name, pos, default=default)
        return self._torch_dim_to_trt_dim(torch_dim)

    def _to_trt_dim(self, dim: int):
        # Helper function, converts one torch dim to trt dim (including negative dim)
        if dim < 0:
            ndim = self.first_input().dim()
            dim = ndim + dim
        return dim - 1 if self.input_has_implicit_batch() else dim

    def _torch_dim_to_trt_dim(self, torch_dim):
        trt_ndim = self.trt_ndim()
        if torch_dim == "_all":
            trt_dim = list(range(trt_ndim))
        elif isinstance(torch_dim, int):
            trt_dim = self._to_trt_dim(torch_dim)
        else:
            trt_dim = [self._to_trt_dim(d) for d in torch_dim]

        assert all(0 <= np.array([trt_dim]) < trt_ndim), "Invalid dimension"
        return trt_dim

    def trt_ndim(self) -> int:
        fi = self.first_input()
        ndim = len(fi._trt.shape)
        assert ndim == self._to_trt_dim(fi.dim()), f"Mismatch torch {tuple(fi.shape)} and trt {fi._trt.shape} dims"
        return ndim

    def input_has_implicit_batch(self):
        # This only happens if the network has implicit batch AND the input is not a constant
        return self.has_implicit_batch() and len(self.first_input().shape) == len(self.first_input()._trt.shape)

    def first_input(self) -> torch.Tensor:
        if self._first_input is not None:
            return self._first_input
        first_input = self.method_args[0]
        while not isinstance(first_input, torch.Tensor):
            first_input = first_input[0]
        self._first_input = first_input
        return first_input

    def get_trt_axes(self, *, trt_dim=__default, torch_dim=__default):
        # Returns an axes bitmask
        assert sum(x is ConversionContext.__default for x in
                   [trt_dim, torch_dim]) <= 1, "Can't have both trt_dim and torch_dim"

        if trt_dim is ConversionContext.__default:
            if torch_dim is ConversionContext.__default:  # Nothing passed, calculate dims
                trt_dim = self.get_trt_dim()
            else:
                trt_dim = self._torch_dim_to_trt_dim(torch_dim)
        if trt_dim == "_all":
            trt_dim = list(range(self.trt_ndim()))
        if isinstance(trt_dim, list):
            trt_dim = tuple(trt_dim)
        if not isinstance(trt_dim, tuple):
            trt_dim = (trt_dim,)

        # create axes bitmask for reduce layer
        axes = 0
        for d in trt_dim:
            assert d >= 0
            axes |= (1 << d)  # don't -1 because it's already trt dim

        return axes

    def get_arg(self, name, pos, default=__default):
        if name in self.method_kwargs:
            return self.method_kwargs[name]
        elif len(self.method_args) > pos:
            return self.method_args[pos]
        elif default is not ConversionContext.__default:
            return default
        else:
            raise ValueError(f"Missing arg {name} at pos {pos}")

    def __enter__(self):
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, type, val, tb):
        for hook in self.hooks:
            hook.__exit__(type, val, tb)

    def add_inputs(self, torch_inputs, input_shapes=None, names=None):
        if names is None:
            names = ['input_%d' % i for i in range(len(torch_inputs))]
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):
            if not hasattr(torch_input, '_trt'):
                shape = tuple(torch_input.shape if input_shapes is None else input_shapes[i])
                if self.has_implicit_batch():
                    shape = shape[1:]
                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=shape,
                    dtype=torch_dtype_to_trt(torch_input.dtype),
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
            array = array.astype(np.int32)
        layer = self.network.add_constant(shape, array)
        return layer.get_output(0)

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
        outputs = method(*args, **kwargs)

        if not skip:
            ctx.setup_method(args, kwargs, outputs, method_str)

            #             print('%s' % (converter.__name__,))
            converter['converter'](ctx)
            if converter['is_real']:
                def _check_shape_recursive(outputs_, pos=()):
                    # Checks for corrupted converter trt tensor outputs, since python api/abi doesn't do so
                    if isinstance(outputs_, torch.Tensor):
                        if hasattr(outputs_, "_trt"):
                            try:
                                len(outputs._trt.shape)
                            except:
                                print(f"Error: bad shape on output {pos} of {method_str}"
                                      f" (expected {tuple(outputs.shape)}")
                        else:
                            print(f"Warning: output {pos} of {method_str} not supported")
                        return
                    for i, output in enumerate(outputs_):
                        _check_shape_recursive(output, pos + (i,))

                _check_shape_recursive(ctx.method_return)

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
