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
        bindings = self.get_bindings({name: value for name, value in
                                      zip(self.input_names, inputs)})

        self.context.execute_v2([t.data_ptr()
                                 if t is not None else 0 for t in bindings])

        outputs = [bindings[self.engine.get_binding_index(oname)] for oname in self.output_names]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

        # bindings = [None] * (len(self.input_names) + len(self.output_names))
        #
        # # create output tensors
        # outputs = []
        # for i, output_name in enumerate(self.output_names):
        #     idx = self.engine.get_binding_index(output_name)
        #     dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
        #     shape = tuple(self.engine.get_binding_shape(idx))
        #     device = torch_device_from_trt(self.engine.get_location(idx))
        #     output = torch.empty(size=shape, dtype=dtype, device=device)
        #     outputs.append(output)
        #     bindings[idx] = output.data_ptr()
        #
        # for i, input_name in enumerate(self.input_names):
        #     idx = self.engine.get_binding_index(input_name)
        #     bindings[idx] = inputs[i].data_ptr()
        #
        # self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        #
        # outputs = tuple(outputs)
        # if len(outputs) == 1:
        #     outputs = outputs[0]
        #
        # return outputs

    def get_bindings(self, input_dict) -> List[Optional[torch.Tensor]]:
        bindings = [None] * self.engine.num_bindings
        # Step 1: shape inputs
        for idx in range(self.engine.num_bindings):
            if self.engine.binding.is_input(idx) and \
                    self.engine.binding.is_shape_binding(idx):
                name = self.engine.get_binding_name(idx)
                print(f"Setting shape input {name}, {idx} to {input_dict[name]}")
                self.context.set_shape_input(idx, input_dict[name])

        assert self.context.all_shape_inputs_specified

        # Step 2: execution bindings (including shape bindings for dynamic inputs)
        for idx in range(self.engine.num_bindings):
            if self.engine.binding.is_input(idx):
                name = self.engine.get_binding_name(idx)
                if -1 in self.engine.get_binding_shape(idx):  # ?? not sure if this is right
                    print(f"Setting binding shape {name}, {idx} to {input_dict[name].shape}")
                    self.context.set_binding_shape(idx, input_dict[name].shape)
                if self.engine.is_execution_binding(idx):
                    bindings[idx] = input_dict[name]

        assert self.context.all_binding_shapes_specified

        # Step 3: output bindings
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding.is_input(idx) and \
                    self.engine.is_execution_binding(idx):
                # name = self.engine.get_binding_name(idx)
                dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                device = torch_device_from_trt(self.engine.get_location(idx))
                shape = self.context.get_binding_shape(idx)  # Infer output shape using TRT
                bindings[idx] = torch.empty(size=shape, dtype=dtype, device=device)

        return bindings

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
