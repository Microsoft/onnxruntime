import onnx
import numpy as np
import os
import sys
import argparse
from onnx import helper, numpy_helper, mapping, utils
from onnx.helper import make_opsetid
from onnx import AttributeProto, SparseTensorProto, TensorProto, GraphProto, ValueInfoProto
import traceback

from typing import Text, Sequence, Any, Optional, Dict, Union, TypeVar, Callable, Tuple, List, cast
from six import text_type, integer_types, binary_type

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, type=str, help="Model file name to save")
    return parser.parse_args()

# This function is now available in ONNX
def make_sparse_tensor_value_info(
        name,  # type: Text
        elem_type,  # type: int
        shape,  # type: Optional[Sequence[Union[Text, int]]]
        doc_string="",  # type: Text
        shape_denotation=None,  # type: Optional[List[Text]]
        ):  # type: (...) -> ValueInfoProto
    """Makes a ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    sparse_tensor_type_proto = value_info_proto.type.sparse_tensor_type
    sparse_tensor_type_proto.elem_type = elem_type

    sparse_tensor_shape_proto = sparse_tensor_type_proto.shape

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        sparse_tensor_shape_proto.dim.extend([])

        if shape_denotation:
            if len(shape_denotation) != len(shape):
                raise ValueError(
                    'Invalid shape_denotation. '
                    'Must be of the same length as shape.')

        for i, d in enumerate(shape):
            dim = sparse_tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, integer_types):
                dim.dim_value = d
            elif isinstance(d, text_type):
                dim.dim_param = d
            else:
                raise ValueError(
                    'Invalid item in shape: {}. '
                    'Needs to of integer_types or text_type.'.format(d))

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return value_info_proto

def create_model(output_file_name):
    matmul_node = helper.make_node("SparseToDenseMatMul", inputs=['sparse_A', 'dense_B'], outputs=['dense_Y'],
                                     name='SpMM', domain='com.microsoft')

    value_info_A = make_sparse_tensor_value_info('sparse_A', TensorProto.FLOAT, [9, 9])
    value_info_B = helper.make_tensor_value_info('dense_B', TensorProto.FLOAT, [9, 9])
    value_info_Y = helper.make_tensor_value_info('dense_Y', TensorProto.FLOAT, [9, 9])

    graph_def = helper.make_graph(nodes=[matmul_node],
                                name='SpMM',
                                inputs=[value_info_A, value_info_B],
                                outputs=[value_info_Y])

    model_def = helper.make_model(graph_def, producer_name='dmitrism', 
        opset_imports=[make_opsetid('com.microsoft', 1)])
        
    onnx.save(model_def, output_file_name)

if __name__ == "__main__":
   try:
      args = parse_arguments()
      sys.exit(create_model(args.output_file))
   except  Exception  as  inst : 
        print("Exception thrown: ", str(inst))
        print(traceback.format_exc())

