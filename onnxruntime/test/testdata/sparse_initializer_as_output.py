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
    parser.add_argument('--node_name', required=True, type=str, help='Constant Node name')
    parser.add_argument("--output_file", required=True, type=str, help="Model file name to save")
 
    args = parser.parse_args()
    return args

# This is availabl in ONNX at a later commit
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


def create_model(constant_node_name, output_file_name):
    dense_shape = [3,3]
    sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
    values_tensor = helper.make_tensor(name='Constant', data_type=TensorProto.FLOAT,
                                       dims=[len(sparse_values)],
                                       vals=np.array(sparse_values).astype(np.float32), raw=False)

    linear_indicies = [2, 3, 5]
    indicies_tensor = helper.make_tensor(name='indicies', data_type=TensorProto.INT64, 
                                       dims=[len(linear_indicies)],
                                       vals=np.array(linear_indicies).astype(np.int64), raw=False)
    sparse_tensor = helper.make_sparse_tensor(values_tensor, indicies_tensor, dense_shape)

    # Nodes
    #sparse_attribute = helper.make_attribute('value', sparse_tensor)
    constant_node = helper.make_node(constant_node_name, inputs=[], outputs=['values'],
                                     name='Constant', domain='', value=sparse_tensor)

     # Outputs, a square matrix
    Values_info = make_sparse_tensor_value_info('values', TensorProto.FLOAT, dense_shape)

    graph_def = helper.make_graph(nodes=[constant_node],
                                  name='ConstantNodeOutput',
                                  inputs=[],
                                  outputs=[Values_info])

    model_def = helper.make_model(graph_def, producer_name='dmitrism', 
      opset_imports=[make_opsetid('', 12)])

    onnx.save(model_def, output_file_name)

if __name__ == "__main__":
   try:
      args = parse_arguments()
      sys.exit(create_model(args.node_name, args.output_file))
   except  Exception  as  inst : 
        print("Exception thrown: ", str(inst))
        print(traceback.format_exc())
