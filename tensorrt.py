# Mock for tensorrt, like pycharm stubs
from enum import Enum
from typing import Callable
import numpy as np


class ILayer:
    def __init__(self):
        self.name = ""
        self.dtype = DataType.FLOAT
        self.inputs = []
        self.kwargs = dict()
        self.output_shape = None
        self.torch_value = None

    def get_input(self, i):
        return self.inputs[i]

    def set_input(self, i, inp):
        while len(self.inputs) <= i:
            self.inputs.append(None)
        self.inputs[i] = inp
        if self.name == "shuffle":
            if i == 1:
                self.reshape_dims = inp
        elif self.name == "concatenation":
            self.reshape_dims = inp
        if self.name == "slice":
            if i == 1:
                self.start = inp
            if i == 2:
                self.shape = inp
            if i == 3:
                self.stride = inp

    def __find_shape(self):
        shape = self.__yolo_shape()
        if self.name in ("constant",):
            shape = self.inputs[1].shape
            self.torch_value = self.inputs[1]
        elif self.name in ("shape",):
            shape = (len(self.inputs[0].shape),)
            self.torch_value = self.inputs[0].shape
        elif self.name in ("slice",):
            shape = self.inputs[2]
            if shape == (1,):  # for slicing a shape tensor
                myslices = [slice(st, st + si, 1 if stride == 0 else stride) for st, si, stride in
                            zip(*self.inputs[1:4])]
                myslices = tuple(myslices) if len(myslices) > 1 else myslices[0]
                self.torch_value = self.inputs[0].torch_value.__getitem__(myslices)
            else:
                shape = shape.torch_value
        elif self.name in ("shuffle",):
            shape = self.inputs[0].shape
            if hasattr(self, "first_transpose"):
                shape = [shape[i] for i in self.first_transpose]
            if hasattr(self, "reshape_dims"):
                newshape = []
                for i, d in enumerate(self.reshape_dims):
                    newshape.append(shape[i] if d == 0 else d)
                shape = newshape
            if hasattr(self, "second_transpose"):
                shape = [shape[i] for i in self.second_transpose]
            if len(shape) <= 1:
                self.torch_value = np.array(self.torch_value).reshape(shape)
        elif self.name in ("gather",):
            shape = list(self.inputs[1].shape)
            shape.append(self.inputs[0].shape[-1])
        elif self.name in ("matrix_multiply",):
            m1, m2 = self.inputs[0], self.inputs[2]
            shape = (*m1.shape[:-1], m2.shape[-1])
        elif self.name in ("unary",):
            shape = self.inputs[0].shape
        elif self.name in ("elementwise",):
            shape = tuple(-1 if -1 in (i0, i1) else max(i0, i1)
                          for i0, i1 in zip(self.inputs[0].shape, self.inputs[1].shape))
        elif self.name in ("reduce",):
            shape = list(self.inputs[0].shape)
            axes = self.inputs[2]
            keep_dims = self.inputs[3]
            assert isinstance(keep_dims, bool)
            axes_list = []
            for i in range(400):
                ij = (1 << i)
                if (axes & ij) != 0:
                    axes_list.append(i)
                if ij > axes:
                    break
            for axis in axes_list:
                shape[axis] = 1 if keep_dims else None
            shape = tuple(d for d in shape if d is not None)
        elif self.name in ("concatenation",):
            cats = self.inputs[0]
            shape = list(cats[0].shape)
            cataxis = [inp.shape[self.axis] for inp in cats]
            if -1 in cataxis:
                shape[self.axis] = -1
            else:
                shape[self.axis] = sum(cataxis)
            if self.axis == 0 and cats[0].shape == (1,):
                self.torch_value = np.concatenate([inp.torch_value for inp in cats], axis=self.axis)
        else:
            print(f"no mock shape for {self.name}")
        assert shape is not None, "shape mocking failed"
        return tuple(map(int, shape))

    def __yolo_shape(self):
        try:
            shape = self.inputs[0].shape
            self.torch_value = self.inputs[0].torch_value
            return shape
        except:
            pass
        try:
            shape = self.inputs[1].shape
            self.torch_value = self.inputs[1].torch_value
            return shape
        except:
            pass
        try:
            shape = self.inputs[0][0].shape
            self.torch_value = self.inputs[0][0].torch_value
            return shape
        except:
            pass
        try:
            shape = self.kwargs["input"].shape
            self.torch_value = self.kwargs["input"]
            return shape
        except:
            pass
        try:
            shape = self.kwargs["inputs"][0].shape
            self.torch_value = self.kwargs["inputs"][0].shape
            return shape
        except:
            pass
        try:
            shape = self.kwargs["tensors"][0].shape
            self.torch_value = self.kwargs["tensors"][0].shape
            return shape
        except:
            pass
        # print("YOLO shape failed")
        return None

    def set_output_type(self, idx, dtype):
        self.dtype = dtype

    def get_output(self, i):
        iten = ITensor()
        iten.shape = self.__find_shape()
        iten.dtype = self.dtype
        iten.torch_value = self.torch_value
        iten.last_layer = self
        # if self.name == "concatenation":

        return iten


class INetworkDefinition:
    def __init__(self):
        self.has_implicit_batch_dimension = False

    def add_input(self, name, dtype, shape):
        ten = ITensor()
        ten.name = name
        ten.dtype = dtype
        ten.shape = shape
        return ten

    def add_loop(self):
        return ILoop()

    def mark_output(self, trt_tensor):
        pass

    def __getattr__(self, name):
        if name[:4] == "add_":
            return make_add_layer_func(name[4:])


def make_add_layer_func(layer_name):
    def add_layer(*inputs, **kwargs):
        layer = ILayer()
        layer.name = layer_name
        layer.inputs = list(inputs) + list(kwargs.values())
        layer.kwargs = kwargs
        return layer

    return add_layer


class ILoop(ILayer):
    def add_trip_limit(self, tensor, kind):
        pass

    def add_iterator(self, tensor):
        return ILayer()


class ITensor:
    def __init__(self):
        # self.name = ""
        self.shape = None
        self.dtype = DataType.HALF
        self.torch_value = None

    def __iter__(self):
        return self.torch_value.__iter__()

    def __len__(self):
        assert len(self.shape) == 1
        return len(self.torch_value)


class IBuilderConfig:
    def set_flag(self, flag):
        pass
    def add_optimization_profile(self, prof):
        pass


class Builder:
    def __init__(self, logger):
        self.logger = logger

    def create_network(self, flags):
        inet = INetworkDefinition()
        inet.has_implicit_batch_dimension = not bool(flags & (1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        return inet

    def create_builder_config(self):
        return IBuilderConfig()

    def build_cuda_engine(self, network: INetworkDefinition):
        raise DeprecationWarning("doesn't work, see c++ docs")

    def build_engine(self, network: INetworkDefinition, config: IBuilderConfig):
        return None

    def create_optimization_profile(self):
        return IOptimizationProfile()


class IOptimizationProfile:
    def set_shape(self, name, min, opt, max):
        pass


class ICudaEngine:
    def __init__(self):
        self.num_bindings = 0


class Logger:
    ERROR = 1
    WARNING = 2

    def __init__(self, min_severity=WARNING):
        self.min_severity = min_severity

    def log(self, severity, msg):
        print(msg)


class NetworkDefinitionCreationFlag:
    EXPLICIT_BATCH = 0


__version__ = "7.0.0.11"


class IInt8Calibrator:
    pass


class CalibrationAlgoType:
    ENTROPY_CALIBRATION_2 = 2


DataType = Enum("DataType", "FLOAT HALF INT8 INT32 BOOL")


class Weights:
    def __init__(self, type_or_a=DataType.FLOAT):
        if isinstance(type_or_a, DataType):
            self.type = type_or_a
            self.a = []
        else:
            self.type = type_or_a.dtype
            self.a = type_or_a


TensorLocation = Enum("TensorLocation", "DEVICE HOST")
BuilderFlag = Enum("BuilderFlag", "GPU_FALLBACK REFIT DEBUG STRICT_TYPES INT8 FP16")
DeviceType = Enum("DeviceType", "GPU DLA")
ElementWiseOperation = Enum("ElementWiseOperation", "SUM SUB DIV MIN MAX POW MUL PROD")
ActivationType = Enum("ActivationType", "TANH RELU")
float32, float16, int8, int32, _ = DataType
UnaryOperation = Enum("UnaryOperation", "EXP LOG SQRT RECIP ABS NEG SIN COS TAN SINH COSH ASIN ACOS ATAN CEIL FLOOR")
ReduceOperation = Enum("ReduceOperation", "MIN MAX PROD AVG SUM")
PoolingType = Enum("PoolingType", "MAX AVERAGE")
TripLimit = Enum("TripLimit", "COUNT WHILE")
ScaleMode = Enum("ScaleMode", "CHANNEL UNIFORM ELEMENTWISE")
MatrixOperation = Enum("MatrixOperation", "NONE VECTOR TRANSPOSE")
