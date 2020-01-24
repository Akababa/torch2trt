from typing import Dict, Optional, List

import tensorrt as trt
import torch
from .conversion_utils import *


class TRTModule(torch.nn.Module):
    def __init__(self, engine: trt.ICudaEngine = None, input_names=None, output_names=None, debug_sync=False):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        self.engine = engine
        self.debug_sync = debug_sync
        if self.engine is not None:
            self.context = self.engine.create_execution_context()
            self.context.debug_sync = debug_sync
            assert self.context.engine is self.engine

        # these must be in order
        self.input_names = input_names
        self.output_names = output_names
        self.bindings = [None] * self.engine.num_bindings
        self.exec_async = True  # slightly faster for some reason

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
            self.context.debug_sync = self.debug_sync
            assert self.context.engine is self.engine

        self.input_names = state_dict[prefix + 'input_names']
        self.output_names = state_dict[prefix + 'output_names']

    def forward(self, *inputs):
        self.set_input_bindings(inputs)

        bindings_ptr = [t.data_ptr()
                        if t is not None else 0 for t in self.bindings]
        if self.exec_async:
            self.context.execute_async_v2(bindings_ptr, torch.cuda.current_stream().cuda_stream)
        else:
            self.context.execute_v2(bindings_ptr)

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
    def reset_bindings(self, input_dict, optimization_profile=0):  # -> List[Optional[torch.Tensor]]:
        self.context.active_optimization_profile = optimization_profile
        if isinstance(input_dict, (list, tuple)):
            input_dict = {k: v for k, v in zip(self.input_names, input_dict)}
        self.bindings = [None] * self.engine.num_bindings
        # Step 1: shape inputs
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx) and \
                    self.engine.is_shape_binding(idx):
                name = self.engine.get_binding_name(idx)
                validate_shape(input_dict[name], self.engine.get_profile_shape(optimization_profile, idx))
                print(f"Setting {name}:{idx} shape input to {input_dict[name]}")
                self.context.set_shape_input(idx, input_dict[name])

        assert self.context.all_shape_inputs_specified

        # Step 2: execution bindings (including shape bindings for dynamic inputs)
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                name = self.engine.get_binding_name(idx)
                validate_shape(input_dict[name].shape, self.engine.get_profile_shape(optimization_profile, idx))
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
                validate_shape(input_dict[name].shape, self.engine.get_profile_shape(optimization_profile, idx))
                dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                device = torch_device_from_trt(self.engine.get_location(idx))
                shape = tuple(self.context.get_binding_shape(idx))  # Infer output shape using TRT
                print(f"Computed {name}:{idx} output shape {shape}")
                self.bindings[idx] = torch.empty(size=shape, dtype=dtype, device=device)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
