import numpy as np
import torch
import tensorrt as trt
from .trt_module import TRTModule
from .calibration import TensorBatchDataset, DatasetCalibrator, DEFAULT_CALIBRATION_ALGORITHM
from .conversion_context import ConversionContext


def torch2trt(module,
              inputs,
              input_names=None,
              optimization_profile=None,  # Required for dynamic input shapes
              output_names=None,
              log_level=trt.Logger.WARNING,
              max_batch_size=1,
              fp16_mode=False,
              max_workspace_size=0,
              strict_type_constraints=False,
              keep_network=True,
              int8_mode=False,
              int8_calib_dataset=None,
              int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
              build_flags=(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)),
              debug_sync=False,
              use_DLA=False,
              ):
    inputs_in = inputs
    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)

    # Make builderconfig
    # Infer input shapes (dynamic) and make optimization profile
    input_shapes = [list(inp.shape) for inp in inputs]
    config = builder.create_builder_config()
    if optimization_profile is not None:
        profile = builder.create_optimization_profile()
        for name, shapelims, inp_shape in zip(input_names, optimization_profile, input_shapes):
            mins, opts, maxs = np.array(shapelims)
            assert all(mins <= opts) and all(opts <= maxs)
            assert all(mins <= inp_shape) and all(inp_shape <= maxs)
            profile.set_shape(name, min=shapelims[0], opt=shapelims[1], max=shapelims[2])
            for d, (mind, maxd) in enumerate(zip(shapelims[0], shapelims[2])):
                if mind != maxd:
                    inp_shape[d] = -1
        config.add_optimization_profile(profile)

    # Set config properties
    config.max_workspace_size = max_workspace_size
    if fp16_mode:
        config.set_flag(trt.BuilderFlags.FP16)
    if debug_sync:
        config.set_flag(trt.BuilderFlags.DEBUG)
    if strict_type_constraints:
        config.set_flag(trt.BuilderFlags.STRICT_TYPES)
    if use_DLA:
        config.default_device_type = trt.DeviceType.DLA

    if int8_mode:
        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)
        config.set_flag(trt.BuilderFlags.INT8)
        # @TODO(jwelsh):  Should we set batch_size=max_batch_size?  Need to investigate memory consumption
        config.int8_calibrator = DatasetCalibrator(inputs, int8_calib_dataset, batch_size=1,
                                                   algorithm=int8_calib_algorithm)

    builder.max_batch_size = max_batch_size

    # Build network

    network = builder.create_network(flags=build_flags)

    with ConversionContext(network) as ctx:
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        ctx.add_inputs(inputs, input_shapes=input_shapes, names=input_names)

        if input_names is None:
            outputs = module(*inputs)
        else:
            outputs = module(**{name: value for name, value in zip(input_names, inputs)})

        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        ctx.mark_outputs(outputs, output_names)

    # Build engine from network and config
    engine = builder.build_engine(network, config)
    assert engine is not None, "build_cuda_network failed"

    module_trt = TRTModule(engine, ctx.input_names, ctx.output_names, debug_sync)

    if keep_network:
        module_trt.network = network

    return module_trt
