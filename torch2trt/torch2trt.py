import torch
import tensorrt as trt
from .trt_module import TRTModule
from .calibration import TensorBatchDataset, DatasetCalibrator, DEFAULT_CALIBRATION_ALGORITHM
from .conversion_context import ConversionContext


def torch2trt(module,
              inputs,
              input_names=None,
              input_shapes=None,  # for dynamic
              output_names=None,
              log_level=trt.Logger.ERROR,
              max_batch_size=1,
              fp16_mode=False,
              max_workspace_size=0,
              strict_type_constraints=False,
              keep_network=True,
              int8_mode=False,
              int8_calib_dataset=None,
              int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
              build_flags=0b0,
              optimization_profile=None):
    inputs_in = inputs

    # copy inputs to avoid modifications to source data
    # inputs = [tensor.clone()[0:1] for tensor in inputs]  # only run single entry ??

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
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

        if not isinstance(outputs, tuple) and not isinstance(outputs, list):
            outputs = (outputs,)
        ctx.mark_outputs(outputs, output_names)

    builder.max_workspace_size = max_workspace_size
    builder.fp16_mode = fp16_mode
    builder.max_batch_size = max_batch_size
    builder.strict_type_constraints = strict_type_constraints

    if int8_mode:

        # default to use input tensors for calibration
        if int8_calib_dataset is None:
            int8_calib_dataset = TensorBatchDataset(inputs_in)

        builder.int8_mode = True

        # @TODO(jwelsh):  Should we set batch_size=max_batch_size?  Need to investigate memory consumption
        builder.int8_calibrator = DatasetCalibrator(inputs, int8_calib_dataset, batch_size=1,
                                                    algorithm=int8_calib_algorithm)

    if optimization_profile is not None:
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        for name, shapelims in optimization_profile.items():
            assert len(shapelims) == 3, "need min, opt, max for optimization profile shapes"
            profile.set_shape(name, *shapelims)
        config.add_optimization_profile(profile)

    engine = builder.build_cuda_engine(network)
    assert engine is not None, "build_cuda_network failed"

    module_trt = TRTModule(engine, ctx.input_names, ctx.output_names)

    if keep_network:
        module_trt.network = network

    return module_trt
