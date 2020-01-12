import tensorrt as trt
import torch
import functools


def postfix_check_output(function):
    print(f"Overwrote {function}")

    @functools.wraps(function)
    def run(*args, **kwargs):
        output = function(*args, **kwargs)
        print(f"Ran {args[0].__class__.__name__}")
        len(output.shape)
        return output

    return run


trt.ILayer.get_output = postfix_check_output(trt.ILayer.get_output)


def torch_dtype_to_trt(dtype):
    if dtype == torch.int8:
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
    if dtype == trt.int8:
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
    if device.type == torch.device('cuda').type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device('cpu').type:
        return trt.TensorLocation.HOST
    else:
        return TypeError('%s is not supported by tensorrt' % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)
