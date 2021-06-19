import onnx
import numpy as np
from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType, attribute_to_kwarg, quantize_nparray


class QPad(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Pad")

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if (self.quantizer.opset_version < 11) or (node.input[0] not in self.quantizer.quantized_value_map):
            super().quantize()
            return
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]

        kwargs = {}
        for attribute in node.attribute:
            kv = attribute_to_kwarg(attribute)
            kwargs.update(kv)

        if 'mode' not in kwargs or kwargs['mode'] == b'constant':
            if len(node.input) > 2:  # There is 3rd input 'constant_value'
                zp_tensor = self.quantizer.model.get_initializer(quantized_input_value.zp_name)
                scale_tensor = self.quantizer.model.get_initializer(quantized_input_value.scale_name)
                if zp_tensor is None or scale_tensor is None:
                    super().quantize()
                    return

                padding_constant_initializer = self.quantizer.model.get_initializer(node.input[2])
                if padding_constant_initializer is not None:
                    zp_array = onnx.numpy_helper.to_array(zp_tensor)
                    zp_value = zp_array.item() if zp_array.ndim == 0 else zp_array[0]
                    scale_array = onnx.numpy_helper.to_array(scale_tensor)
                    scale_value = scale_array.item() if scale_array.ndim == 0 else scale_array[0]
                    padding_constant_array = onnx.numpy_helper.to_array(padding_constant_initializer)
                    quantized_padding_constant_array = quantize_nparray(self.quantizer.input_qType,
                                                                        padding_constant_array, scale_value, zp_value)
                    quantized_padding_constant_name = node.input[2] + "_quantized"
                    quantized_padding_constant_initializer = onnx.numpy_helper.from_array(
                        quantized_padding_constant_array, quantized_padding_constant_name)
                    # Suppose this padding constant initializer only used by the node
                    self.quantizer.model.remove_initializer(padding_constant_initializer)
                    self.quantizer.model.add_initializer(quantized_padding_constant_initializer)
                    node.input[2] = quantized_padding_constant_name
                else:
                    # TODO: check quantize_inputs after sub graph is supported
                    pad_value_qnodes = self.quantizer._get_quantize_input_nodes(node, 2, self.quantizer.input_qType,
                                                                                quantized_input_value.scale_name,
                                                                                quantized_input_value.zp_name)
                    self.quantizer.new_nodes += [pad_value_qnodes]
                    node.input[2] = pad_value_qnodes.output[0]
            else:
                node.input.extend([quantized_input_value.zp_name])  # pad zero_point for original zero

        # Create an entry for output quantized value
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, quantized_input_value.zp_name,
                                                QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += [node]
