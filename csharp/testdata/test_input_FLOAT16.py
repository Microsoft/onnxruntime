# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import helper
from onnx.helper import make_opsetid
from onnx import TensorProto

input_info = helper.make_tensor_value_info('input', TensorProto.FLOAT16, [1, 5])
output_info = helper.make_tensor_value_info('output', TensorProto.FLOAT16, [1, 5])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'Slice',  # node name
    ['input'],  # inputs
    ['output'],  # outputs
    axes=[0, 1],  # attributes
    ends=[1, 5],
    starts=[0, 0]
)

graph_def = helper.make_graph(nodes=[node_def], name='test_input_FLOAT16',
                              inputs=[input_info], outputs=[output_info])

model_def = helper.make_model(graph_def, producer_name='AIInfra',
                              opset_imports=[make_opsetid('', 7)])

onnx.checker.check_model(model_def)
onnx.helper.strip_doc_string(model_def)
final_model = onnx.shape_inference.infer_shapes(model_def)
onnx.checker.check_model(final_model)
onnx.save(final_model, 'test_types_FLOAT16.onnx')
