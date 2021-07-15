import onnx
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain
from onnx import onnx_pb as onnx_proto
'''
    Quantize Attention
'''


class AttentionQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        '''
            parameter node: Attention node.
            parameter new_nodes_list: List of new nodes created before processing this node.
            return: a list of nodes in topological order that represents quantized Attention node.
        '''
        node = self.node
        assert (node.op_type == "Attention")

        (quantized_input_names, zero_point_names, scale_names, nodes) = \
            self.quantizer.quantize_inputs(node, [0, 1], reduce_range=True, op_level_per_channel=True)

        qattention_name = "" if node.name == "" else node.name + "_quant"

        inputs = []
        inputs.extend(quantized_input_names)
        inputs.extend([node.input[2]])
        inputs.extend(scale_names)
        inputs.extend([node.input[3] if len(node.input) > 3 else ""])
        inputs.extend(zero_point_names)
        inputs.extend([node.input[4] if len(node.input) > 4 else ""])
        inputs.extend([node.input[5] if len(node.input) > 5 else ""])

        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        qattention_node = onnx.helper.make_node("QAttention", inputs, node.output, qattention_name, **kwargs)
        nodes.append(qattention_node)

        self.quantizer.new_nodes += nodes
