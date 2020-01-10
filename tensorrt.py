from enum import Enum
from typing import Callable


class ILayer:
    def __init__(self):
        self.name = ""
        self.dtype = None
        self.inputs = []
        self.output_shape = None

    def get_input(self, i):
        return self.inputs[i]

    def get_output(self, i):
        iten = ITensor()
        if self.output_shape is not None:
            iten.shape = self.output_shape
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

            def add_layer(*inputs, **kwargs):
                layer = ILayer()
                layer.inputs = inputs + tuple(kwargs.values())
                if name[4:] in ("elementwise", "scale"):
                    layer.output_shape = inputs[0].shape
                else:
                    try:
                        layer.output_shape = inputs[0].shape
                        layer.output_shape = kwargs["inputs"][0].shape
                    except:
                        pass
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
        self.shape = (0,)
        self.dtype = DataType.HALF


class Builder:
    def __init__(self, logger):
        self.logger = logger

    def create_network(self, flags):
        inet = INetworkDefinition()
        inet.has_implicit_batch_dimension = bool(flags & (1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        return inet

    def create_builder_config(self):
        return IBuilderConfig()

    def build_cuda_engine(self, network: INetworkDefinition):
        return None

    def create_optimization_profile(self):
        return IOptimizationProfile()


class IOptimizationProfile:
    def set_shape(self, name, shape):
        pass


class IBuilderConfig:
    def add_optimization_profile(self, prof):
        pass


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


class TensorLocation:
    DEVICE = 0
    HOST = 1


DataType = Enum("DataType", "FLOAT HALF INT8 INT32 BOOL")


class Weights:
    def __init__(self, type_or_a=DataType.FLOAT):
        if isinstance(type_or_a, DataType):
            self.type = type_or_a
            self.a = []
        else:
            self.type = type_or_a.dtype
            self.a = type_or_a


ElementWiseOperation = Enum("ElementWiseOperation", "SUM SUB DIV MIN MAX POW MUL PROD")
ActivationType = Enum("ActivationType", "TANH RELU")
float32, float16, int8, int32, _ = DataType
UnaryOperation = Enum("UnaryOperation", "EXP LOG SQRT RECIP ABS NEG SIN COS TAN SINH COSH ASIN ACOS ATAN CEIL FLOOR")
ReduceOperation = Enum("ReduceOperation", "MIN MAX PROD AVG SUM")
PoolingType = Enum("PoolingType", "MAX AVERAGE")
TripLimit = Enum("TripLimit", "COUNT WHILE")
ScaleMode = Enum("ScaleMode", "CHANNEL UNIFORM ELEMENTWISE")
