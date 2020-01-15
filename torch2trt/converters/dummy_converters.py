import torch
from ..conversion_context import *


def is_private(method):
    method = method.split('.')[-1]  # remove prefix
    return method[0] == '_' and method[1] is not '_'


def is_function_type(method):
    fntype = eval(method + '.__class__.__name__')
    return fntype == 'function' or fntype == 'builtin_function_or_method' or fntype == 'method_descriptor'


def get_methods(namespace):
    methods = []
    for method in dir(eval(namespace)):
        full_method = namespace + '.' + method
        if not is_private(full_method) and is_function_type(full_method):
            methods.append(full_method)
    return methods


TORCH_METHODS = []
TORCH_METHODS += get_methods('torch')
TORCH_METHODS += get_methods('torch.Tensor')
TORCH_METHODS += get_methods('torch.nn.functional')

for method in TORCH_METHODS:
    @tensorrt_converter(method, is_real=False)
    def warn_method(ctx):
        # THIS will show up when using DEBUGGER, ignore it
        print('Warning: Encountered known unsupported method %s' % ctx.method_str)


@tensorrt_converter('torch.Tensor.__repr__', is_real=False)
def warn_method(ctx):
    # THIS will show up when using DEBUGGER, ignore it
    pass
