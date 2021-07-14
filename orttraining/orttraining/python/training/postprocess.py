import sys
import os.path
from onnx import *
import onnx
import numpy as np
import struct

from onnx import helper
from onnx import numpy_helper


def run_postprocess(model):
    # this post pass is not required for pytorch >= 1.5
    # where add_node_name in torch.onnx.export is default to True
    model = add_name(model)

    # this post pass is not required for pytorch > 1.6
    model = fuse_softmaxNLL_to_softmaxCE(model)

    model = fix_expand_shape(model)
    model = fix_expand_shape_pt_1_5(model)
    return model

def find_input_node(model, arg):
    result = []
    for node in model.graph.node:
        for output in node.output:
            if output == arg:
                result.append(node)
    return result[0] if len(result)== 1 else None

def find_output_node(model, arg):
    result = []
    for node in model.graph.node:
        for input in node.input:
            if input == arg:
                result.append(node)
    return result[0] if len(result) == 1 else result

def add_name(model):
    i = 0
    for node in model.graph.node:
       node.name = '%s_%d' %(node.op_type, i)
       i += 1
    return model

# Expand Shape PostProcess

def fix_expand_shape(model):
    expand_nodes = [n for n in model.graph.node if n.op_type == 'Expand']
    model_inputs_names = [i.name for i in model.graph.input]

    for expand_node in expand_nodes:
        shape = find_input_node(model, expand_node.input[1])
        if shape.op_type == 'Shape':
            # an expand subgraph
            # Input    Input2
            # |        |
            # |        Shape
            # |        |
            # |__    __|
            #    |  |
            #   Expand
            #     |
            #   output
            #
            # Only if Input2 is one of the model inputs, assign Input2's shape to output of expand.
            shape_input_name = shape.input[0]
            if shape_input_name in model_inputs_names:
                index = model_inputs_names.index(shape_input_name)
                expand_out = model.graph.value_info.add()
                expand_out.name = expand_node.output[0]
                expand_out.type.CopyFrom(model.graph.input[index].type)
    return model

def fix_expand_shape_pt_1_5(model):
    # expand subgraph
    #                      Constant
    #                        +
    #                     ConstantOfShape
    #                      | +  |
    #                      | +  |
    # (Reshape subgraph)   Mul  |
    #       |___   _________|   |
    #       +   | |             |
    #       +  Equal            |
    #       +++++|++++++++++++++|++
    #            |____________  | +
    #                         | | +
    #   (subgraph)            Where
    #       |                   |
    #       |_____   ___________|
    #             | |
    #           Expand
    #             |
    #           output
    #
    # where the Reshape subgraph is
    #
    #  Input
    #   | |
    #   | |___________________
    #   |                     |
    #  Shape   Constant      Shape   Constant
    #   |  ______|            |  ______|
    #   | |                   | |
    #  Gather                Gather
    #   |                     |
    # Unsqueeze             Unsqueeze
    #   |                     |
    #   |  ..Number of dims.. |
    #   |    _________________|
    #   |...|
    #  Concat                       Constant
    #     |                            |
    #     |______    __________________|
    #            |  |
    #           Reshape
    #             |
    #           output
    #
    # This pass will copy Input's shape to the output of Expand.
    expand_nodes = [n for n in model.graph.node if n.op_type == 'Expand']
    model_inputs_names = [i.name for i in model.graph.input]

    for expand_node in expand_nodes:
        n_where = find_input_node(model, expand_node.input[1])
        if n_where.op_type != 'Where':
            continue

        n_equal = find_input_node(model, n_where.input[0])
        n_cos = find_input_node(model, n_where.input[1])
        n_reshape = find_input_node(model, n_where.input[2])

        if n_equal.op_type != 'Equal' or n_cos.op_type != 'ConstantOfShape' or n_reshape.op_type != 'Reshape':
            continue

        n_reshape_e = find_input_node(model, n_equal.input[0])
        n_mul = find_input_node(model, n_equal.input[1])
        if n_reshape_e != n_reshape or n_mul.op_type != 'Mul':
            continue

        n_cos_m = find_input_node(model, n_mul.input[0])
        n_constant = find_input_node(model, n_mul.input[1])
        if n_cos_m != n_cos or n_constant.op_type != 'Constant':
            continue

        n_concat = find_input_node(model, n_reshape.input[0])
        n_constant_r = find_input_node(model, n_reshape.input[1])
        if n_concat.op_type != 'Concat' or n_constant_r.op_type != 'Constant':
            continue

        n_input_candidates = []
        for concat_in in n_concat.input:
            n_unsqueeze = find_input_node(model, concat_in)
            if n_unsqueeze.op_type != 'Unsqueeze':
                break
            n_gather = find_input_node(model, n_unsqueeze.input[0])
            if n_gather.op_type != 'Gather':
                break
            n_shape = find_input_node(model, n_gather.input[0])
            n_constant_g = find_input_node(model, n_gather.input[1])
            if n_shape.op_type != 'Shape' or n_constant_g.op_type != 'Constant':
                break
            n_input = n_shape.input[0]
            if not n_input in model_inputs_names:
                break
            n_input_candidates.append(n_input)

        if not n_input_candidates or not all(elem == n_input_candidates[0] for elem in n_input_candidates):
            continue

        index = model_inputs_names.index(n_input_candidates[0])
        expand_out = model.graph.value_info.add()
        expand_out.name = expand_node.output[0]
        expand_out.type.CopyFrom(model.graph.input[index].type)
    return model

# LayerNorm PostProcess

def find_nodes(graph, op_type):
    nodes = []
    for node in graph.node:
        if node.op_type == op_type:
            nodes.append(node)
    return nodes

def is_type(node, op_type):
    if node is None or isinstance(node, list):
        return False
    return node.op_type == op_type

def add_const(model, name, output, t_value = None, f_value = None):
    const_node = model.graph.node.add()
    const_node.op_type = 'Constant'
    const_node.name = name
    const_node.output.extend([output])
    attr = const_node.attribute.add()
    attr.name = 'value'
    if t_value is not None:
        attr.type = 4
        attr.t.CopyFrom(t_value)
    else:
        attr.type = 1
        attr.f = f_value
    return const_node

def layer_norm_transform(model):
    # DEPRECATED: This pass is no longer needed as the transform is handled at the backend.
    # Converting below subgraph
    #
    # input
    #   |
    # ReduceMean
    #   |
    #  Sub                         Constant
    #  _||_____                       |
    # |        |                      |
    # |        |                      |
    # |   (optional) Cast      (optional) Cast
    # |        |                      |
    # |        |  ____________________|
    # |        | |
    # |        Pow
    # |        |
    # |       ReduceMean
    # |        |
    # |        Add
    # |        |
    # |__    __Sqrt
    #    |  |
    #     Div  (weight)
    #     |       |
    #     |  _____|
    #     | |
    #     Mul   (bias)
    #     |       |
    #     |  _____|
    #     | |
    #     Add
    #     |
    #     output
    #
    # to the below subgraph
    #
    # input    (weight)    (bias)
    #   |         |          |
    #   |  _______|          |
    #   | |  ________________|
    #   | | |
    # LayerNormalization
    #   |
    # output
    graph = model.graph

    nodes_ReduceMean = find_nodes(graph, "ReduceMean")

    id = 0
    layer_norm_nodes = []
    remove_nodes = []
    for reduce_mean in nodes_ReduceMean:
        # check that reduce_mean output is Sub
        sub = find_output_node(model, reduce_mean.output[0])
        if not is_type(sub, "Sub"):
            continue

        # check that sub output[0] is Div and output[1] is Pow
        pow, div = find_output_node(model, sub.output[0])
        if is_type(pow, "Cast"):
            # During an update in PyTorch, Cast nodes are inserted between Sub and Pow.
            remove_nodes += [pow]
            pow = find_output_node(model, pow.output[0])
            if not is_type(pow, "Pow"):
                continue
            cast_pow = find_input_node(model, pow.input[1])
            if not is_type(cast_pow, "Cast"):
                continue
            remove_nodes += [cast_pow]
        if not is_type(div, "Div") or not is_type(pow, "Pow"):
            continue

        # check that pow ouput is ReduceMean
        reduce_mean2 = find_output_node(model, pow.output[0])
        if not is_type(reduce_mean2, "ReduceMean"):
            continue

        # check that reduce_mean2 output is Add
        add = find_output_node(model, reduce_mean2.output[0])
        if not is_type(add, "Add"):
            continue

        # check that add output is Sqrt
        sqrt = find_output_node(model, add.output[0])
        if not is_type(sqrt, "Sqrt"):
            continue

        # check that sqrt output is div
        if div != find_output_node(model, sqrt.output[0]):
            continue

        # check if div output is Mul
        optional_mul = find_output_node(model, div.output[0])
        if not is_type(optional_mul, "Mul"):
            optional_mul = None
            continue # default bias and weight not supported

        # check if mul output is Add
        if optional_mul is not None:
            optional_add = find_output_node(model, optional_mul.output[0])
        else:
            optional_add = find_output_node(model, div.output[0])
        if not is_type(optional_add, "Add"):
            optional_add = None
            continue # default bias and weight not supported


        # add nodes to remove_nodes
        remove_nodes.extend([reduce_mean, sub, div, pow, reduce_mean2, add, sqrt])

        # create LayerNorm node
        layer_norm_input = []
        layer_norm_output = []

        layer_norm_input.append(reduce_mean.input[0])

        if optional_mul is not None:
            remove_nodes.append(optional_mul)
            weight = optional_mul.input[1]
            layer_norm_input.append(weight)

        if optional_add is not None:
            remove_nodes.append(optional_add)
            bias = optional_add.input[1]
            layer_norm_input.append(bias)

        if optional_add is not None:
            layer_norm_output.append(optional_add.output[0])
        elif optional_mul is not None:
            layer_norm_output.append(optional_mul.output[0])
        else:
            layer_norm_output.append(div.output[0])

        layer_norm_output.append('saved_mean_' + str(id))
        layer_norm_output.append('saved_inv_std_' + str(id))

        epsilon_node = find_input_node(model, add.input[1])
        epsilon = epsilon_node.attribute[0].t.raw_data
        epsilon = struct.unpack('f', epsilon)[0]

        layer_norm = helper.make_node("LayerNormalization",
                                      layer_norm_input,
                                      layer_norm_output,
                                      "LayerNormalization_" + str(id),
                                      None,
                                      axis = reduce_mean.attribute[0].ints[0],
                                      epsilon = epsilon)
        layer_norm_nodes.append(layer_norm)
        id += 1

    # remove orphan constant nodes
    for constant in graph.node:
        if constant.op_type == "Constant" and constant not in remove_nodes:
            is_orphan = True
            for out_name in constant.output:
                out = find_output_node(model, out_name)
                if out not in remove_nodes:
                    is_orphan = False
            if is_orphan:
                remove_nodes.append(constant)

    all_nodes = []
    for node in graph.node:
        if node not in remove_nodes:
            all_nodes.append(node)

    for node in layer_norm_nodes:
        all_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(all_nodes)
    return model

# Fuse SoftmaxCrossEntropy

def fuse_softmaxNLL_to_softmaxCE(onnx_model):
    # Converting below subgraph
    #
    #    (subgraph)
    #        |
    #    LogSoftmax     (target)    (optional weight)
    #        |             |             |
    #   nll_loss/NegativeLogLikelihoodLoss
    #                   |
    #                output
    #
    # to the following
    #
    #    (subgraph)     (target)    (optional weight)
    #        |             |        _____|
    #        |             |       |
    #       SparseSoftmaxCrossEntropy
    #                   |
    #                output
    nll_count = 0
    while True:
        nll_count = nll_count + 1
        nll_loss_node = None
        nll_loss_node_index = 0
        for nll_loss_node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "nll_loss" or node.op_type == "NegativeLogLikelihoodLoss":
                nll_loss_node = node
                break

        if nll_loss_node is None:
            break

        softmax_node = None
        softmax_node_index = 0
        label_input_name = None
        weight_input_name = None
        for softmax_node_index, node in enumerate(onnx_model.graph.node):
            if node.op_type == "LogSoftmax":
                # has to be connected to nll_loss
                if len(nll_loss_node.input) > 2:
                    weight_input_name = nll_loss_node.input[2]
                if node.output[0] == nll_loss_node.input[0]:
                    softmax_node = node
                    label_input_name = nll_loss_node.input[1]
                    break
                elif node.output[0] == nll_loss_node.input[1]:
                    softmax_node = node
                    label_input_name = nll_loss_node.input[0]
                    break
            else:
                if softmax_node is not None:
                    break

        if softmax_node is None:
            break

        # delete nll_loss and LogSoftmax nodes in order
        if nll_loss_node_index < softmax_node_index:
            del onnx_model.graph.node[softmax_node_index]
            del onnx_model.graph.node[nll_loss_node_index]
        else:
            del onnx_model.graph.node[nll_loss_node_index]
            del onnx_model.graph.node[softmax_node_index]

        probability_output_name = softmax_node.output[0]
        node = onnx_model.graph.node.add()
        inputs = [softmax_node.input[0], label_input_name, weight_input_name] if weight_input_name else [softmax_node.input[0], label_input_name]
        node.CopyFrom(onnx.helper.make_node("SparseSoftmaxCrossEntropy", inputs,
                                            [nll_loss_node.output[0], probability_output_name],
                                            "nll_loss_node_" + str(nll_count)))

    return onnx_model
