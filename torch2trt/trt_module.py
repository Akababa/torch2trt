from typing import Dict, Optional, List

import tensorrt as trt
import torch
from .conversion_utils import *


def trt_num_inputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            count += 1
    return count


def trt_num_outputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            count += 1
    return count


class TRTModule(torch.nn.Module):
    def __init__(self, engine: trt.ICudaEngine = None, input_names=None, output_names=None, debug_sync=False):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
            self.context.debug_sync = debug_sync

        # these must be in order
        self.input_names = input_names
        self.output_names = output_names
        self.bindings = [None] * self.engine.num_bindings

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + 'engine'] = bytearray(self.engine.serialize())
        state_dict[prefix + 'input_names'] = self.input_names
        state_dict[prefix + 'output_names'] = self.output_names

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        engine_bytes = state_dict[prefix + 'engine']

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            self.context = self.engine.create_execution_context()

        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']

    def forward(self, *inputs):
        self.set_input_bindings(inputs)

        self.context.execute_v2([t.data_ptr()
                                 if t is not None else 0 for t in self.bindings])

        outputs = [self.bindings[self.engine.get_binding_index(oname)] for oname in self.output_names]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # Can't call this before reset_bindings
    def set_input_bindings(self, input_dict):
        if isinstance(input_dict, (list, tuple)):
            input_dict = {k: v for k, v in zip(self.input_names, input_dict)}
        # Step 2: execution bindings (including shape bindings for dynamic inputs)
        for idx in range(self.engine.num_bindings):
            if self.engine.is_execution_binding(idx):
                # print(f"Setting execution binding {name}, {idx}")
                if self.engine.binding_is_input(idx):
                    name = self.engine.get_binding_name(idx)
                    self.bindings[idx] = input_dict[name]

        # assert self.context.all_binding_shapes_specified

    # After calling this once, only set_input_bindings is needed on forward passes
    def reset_bindings(self, input_dict):  # -> List[Optional[torch.Tensor]]:
        if isinstance(input_dict, (list, tuple)):
            input_dict = {k: v for k, v in zip(self.input_names, input_dict)}
        self.bindings = [None] * self.engine.num_bindings
        # Step 1: shape inputs
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx) and \
                    self.engine.is_shape_binding(idx):
                name = self.engine.get_binding_name(idx)
                print(f"Setting {name}:{idx} shape input to {input_dict[name]}")
                self.context.set_shape_input(idx, input_dict[name])

        assert self.context.all_shape_inputs_specified

        # Step 2: execution bindings (including shape bindings for dynamic inputs)
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                if -1 in self.engine.get_binding_shape(idx):  # ?? not sure if this is right
                    print(f"Setting {name}:{idx} binding shape to {input_dict[name].shape}")
                    self.context.set_binding_shape(idx, input_dict[name].shape)
                if self.engine.is_execution_binding(idx):
                    print(f"Setting {name}:{idx} execution binding (input)")
                    self.bindings[idx] = input_dict[name]

        assert self.context.all_binding_shapes_specified

        # Step 3: output bindings
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx) and \
                    self.engine.is_execution_binding(idx):
                name = self.engine.get_binding_name(idx)
                dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                device = torch_device_from_trt(self.engine.get_location(idx))
                shape = tuple(self.context.get_binding_shape(idx))  # Infer output shape using TRT
                print(f"Computed {name}:{idx} output shape {shape}")
                self.bindings[idx] = torch.empty(size=shape, dtype=dtype, device=device)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
