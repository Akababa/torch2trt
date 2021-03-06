# Mock for tensorrt, like pycharm stubs
from enum import Enum
from typing import Callable, Tuple
import numpy as np

_bool = __builtins__["bool"]  # since I shadowed it


class ILayer:
    def __init__(self):
        self.name = ""
        self.opname = ""
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
        if self.opname == "shuffle":
            if i == 1:
                self.reshape_dims = inp
        elif self.opname == "concatenation":
            self.reshape_dims = inp
        if self.opname == "slice":
            if i == 1:
                self.start = inp
            if i == 2:
                self.shape = inp
            if i == 3:
                self.stride = inp

    def set_output_dtype(self, i, dtype):
        self.dtype = dtype

    def __find_shape(self):
        shape = self.__yolo_shape()
        if self.opname in ("constant",):
            shape = self.inputs[0]
            self.torch_value = self.inputs[1].a if isinstance(self.inputs[1], Weights) else self.inputs[1]
        elif self.opname in ("shape",):
            shape = (len(self.inputs[0].shape),)
            self.torch_value = self.inputs[0].shape
        elif self.opname in ("slice",):
            shape = self.inputs[2]
            if "[Shape]_output" in self.inputs[0].name:  # for slicing a shape tensor
                myslices = [slice(st, st + si, 1 if stride == 0 else stride) for st, si, stride in
                            zip(*self.inputs[1:4])]
                myslices = tuple(myslices) if len(myslices) > 1 else myslices[0]
                self.torch_value = self.inputs[0].torch_value.__getitem__(myslices)
            # else:
            #     shape = shape.torch_value
        elif self.opname in ("shuffle",):
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
        elif self.opname in ("gather",):
            shape = list(self.inputs[0].shape)
            shape.pop(self.inputs[2])
            shapeidx = list(self.inputs[1].shape)
            # shapeidx.pop()
            shapeidx.extend(shape)
            shape = shapeidx
        elif self.opname in ("matrix_multiply",):
            m1, m2 = self.inputs[0], self.inputs[2]
            shape = (*m1.shape[:-1], m2.shape[-1])
        elif self.opname in ("unary",):
            shape = self.inputs[0].shape
        elif self.opname in ("elementwise",):
            shape = tuple(-1 if -1 in (i0, i1) else max(i0, i1)
                          for i0, i1 in zip(self.inputs[0].shape, self.inputs[1].shape))
            if shape == ():
                v1, v2 = int(self.inputs[0].torch_value), int(self.inputs[1].torch_value)
                if -1 in (v1, v2):
                    self.torch_value = -1
                elif self.inputs[2] == ElementWiseOperation.SUM:
                    self.torch_value = v1 + v2
                elif self.inputs[2] == ElementWiseOperation.SUB:
                    self.torch_value = v1 - v2
                else:
                    print("couldn't infer shape dim")
        elif self.opname in ("reduce",):
            shape = list(self.inputs[0].shape)
            axes = self.inputs[2]
            keep_dims = self.inputs[3]
            assert isinstance(keep_dims, _bool)
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
        elif self.opname in ("concatenation",):
            cats = self.inputs[0]
            shape = list(cats[0].shape)
            cataxis = [inp.shape[self.axis] for inp in cats]
            if -1 in cataxis:
                shape[self.axis] = -1
            else:
                shape[self.axis] = sum(cataxis)
            if self.axis == 0 and len(cats[0].shape) == 1:
                self.torch_value = np.concatenate([inp.torch_value for inp in cats], axis=self.axis)
        elif self.opname == "fully_connected":
            shape = list(self.inputs[0].shape[:-3]) + [self.inputs[1], 1, 1]
            # shape[-3] = self.inputs[2].shape[0]
        elif self.opname == "select":
            shape = self.inputs[1].shape
        elif self.opname in ["identity", "activation"]:
            shape = self.inputs[0].shape
            if hasattr(self, "precision"):
                self.dtype = self.precision
        elif self.opname in ["activation"]:
            shape = self.inputs[0].shape
        else:
            print(f"no mock shape for {self.opname}")
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
        iten.name = f"({self.name}) [{self.opname[0].upper()}{self.opname[1:]}]_output"

        return iten


class INetworkDefinition:
    def __init__(self):
        self.has_implicit_batch_dimension = False
        self._layers = []
        self.num_layers = 0

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

    def __getattr__(self, name) -> Callable[..., ILayer]:
        if name[:4] != "add_":
            raise AttributeError

        def add_layer(*inputs, **kwargs) -> ILayer:
            layer = ILayer()
            layer.name = f"Unnamed Layer* {self.num_layers}"
            self.num_layers += 1
            layer.opname = name[4:]
            layer.inputs = list(inputs) + list(kwargs.values())
            layer.kwargs = kwargs
            self._layers.append(layer)
            return layer

        return add_layer


class ILoop(ILayer):
    def add_trip_limit(self, tensor, kind):
        pass

    def add_iterator(self, tensor):
        return ILayer()


class ITensor:
    shape: Tuple[int]

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
        inet.has_implicit_batch_dimension = not \
            _bool(flags & (1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
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
    INFO = 3
    VERBOSE = 4

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
ElementWiseOperation = Enum("ElementWiseOperation",
                            "EQUAL DIV SUB POW LESS OR MIN FLOOR_DIV GREATER XOR MAX AND PROD SUM")
ActivationType = Enum("ActivationType", "TANH RELU")
float32, float16, int8, int32, bool = DataType
UnaryOperation = Enum("UnaryOperation", "EXP LOG SQRT RECIP ABS NEG SIN COS TAN SINH COSH ASIN ACOS ATAN CEIL FLOOR")
ReduceOperation = Enum("ReduceOperation", "MIN MAX PROD AVG SUM")
PoolingType = Enum("PoolingType", "MAX AVERAGE")
TripLimit = Enum("TripLimit", "COUNT WHILE")
ScaleMode = Enum("ScaleMode", "CHANNEL UNIFORM ELEMENTWISE")
MatrixOperation = Enum("MatrixOperation", "NONE VECTOR TRANSPOSE")
