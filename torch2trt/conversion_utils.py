import numpy as np

import tensorrt as trt
import torch
import functools

get_output_old = trt.ILayer.get_output


# Wrap tensorrt.ILayer.get_output for better error checking and debugging
@functools.wraps(get_output_old)
def __get_output_v2(*args, **kwargs):
    # allargs = tuple(args) + tuple(kwargs.values())
    # print(allargs)
    output = get_output_old(*args, **kwargs)
    print(f"+ \"{output.name}\" with shape {output.shape}, dtype {output.dtype}")
    assert output.shape.__len__() >= 0, "Invalid ILayer inputs"
    return output


def wrap_get_output():
    trt.ILayer.get_output = __get_output_v2
    # print(f"Wrapped {get_output_old} - This should only be called once")


def unwrap_get_output():
    trt.ILayer.get_output = get_output_old


def torch_dtype_to_trt(dtype):
    assert isinstance(dtype, torch.dtype)
    if dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError('%s is not supported by tensorrt' % dtype)


def torch_dtype_from_trt(dtype):
    assert isinstance(dtype, trt.DataType)
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_to_trt(device):
    assert isinstance(device, torch.device)
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        print("WARNING: on cpu, use model.to(device='cuda') before calling torch2trt")
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    assert isinstance(device, trt.TensorLocation)
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def validate_shape(shape, minoptmax):
    mins, opts, maxs = np.array(minoptmax)
    shape = np.array(shape)
    assert all(mins <= opts), f"{list(mins)} <= {list(opts)} not satisfied"
    assert all(opts <= maxs), f"{list(opts)} <= {list(maxs)} not satisfied"
    assert all(mins <= shape), f"{list(mins)} <= {list(shape)} not satisfied"
    assert all(shape <= maxs), f"{list(shape)} <= {list(maxs)} not satisfied"
