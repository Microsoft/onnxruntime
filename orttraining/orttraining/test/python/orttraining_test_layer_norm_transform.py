import onnx

def find_node(graph_proto, op_type):
    nodes = []
    map_input_node = {}
    for node in graph_proto.node:
        if node.op_type == op_type:
            map_input_node[node.input[0]] = node
            if op_type == 'Div' or op_type == 'Mul':
                map_input_node[node.input[1]] = node
            nodes.append(node)
    return nodes, map_input_node

def gen_attribute(key, value):
    attr = AttributeProto()
    attr.name = key
    attr.ints.extend(int(v) for v in value)
    attr.type = AttributeProto.INTS
    return attr

def layer_norm_transform(model_proto):
    # a layer norm subgraph
    # input
    #   |
    # ReduceMean
    #  __|____
    # |       |
    # Sub     Sub
    # |       |
    # |       Pow
    # |        |
    # |        ReduceMean
    # |        |
    # |        Add
    # |        |
    # |__    __Sqrt
    #    |  |
    #     Div
    #     |
    #     Mul
    #     |
    #     Add
    #     |
    #     output

    graph_proto = model_proto.graph

    _,  map_input_Div = find_node(graph_proto, 'Div')

    _,  map_input_Sqrt = find_node(graph_proto, 'Sqrt')

    _,  map_input_Add = find_node(graph_proto, 'Add')

    nodes_ReduceMean,  map_input_ReduceMean = find_node(graph_proto, 'ReduceMean')

    _,  map_input_Pow = find_node(graph_proto, 'Pow')

    _,  map_input_Mul = find_node(graph_proto, 'Mul')

    # find right side Sub (see the layer norm subgrapg)
    nodes_Sub = []
    map_input_Sub = {}
    for node in graph_proto.node:
        if node.op_type == 'Sub':
            if node.output[0] in map_input_Pow:
                nodes_Sub.append(node)
                map_input_Sub[node.input[1]] = node

    # find first ReduceMean
    first_ReduceMean = []
    first_ReduceMean_outputs = []
    for node in nodes_ReduceMean:
        if node.output[0] in map_input_Sub:
            first_ReduceMean.append(node)
            first_ReduceMean_outputs.append(node.output[0])

    # find constant node
    nodes_Constant = []
    map_output_Constant = {}
    for node in graph_proto.node:
        if node.op_type == 'Constant':
            nodes_Constant.append(node)
            map_output_Constant[node.output[0]] = node

    id = 0
    removed_nodes = []
    layer_norm_nodes = []
    # Replace with layer norm
    for node in first_ReduceMean:
        layer_norm_input = []
        layer_norm_output = []
        layer_norm_input.append(node.input[0])

        # collect nodes within a layer norm subgraph.
        # skip building layer norm node if there is a pattern miss-match.
        if node.output[0] not in map_input_Sub:
            continue

        node_sub = map_input_Sub[node.output[0]]
        if node_sub.output[0] not in map_input_Pow:
            continue

        node_pow = map_input_Pow[node_sub.output[0]]
        if node_pow.output[0] not in map_input_ReduceMean:
            continue

        node_reduce = map_input_ReduceMean[node_pow.output[0]]
        if node_reduce.output[0] not in map_input_Add:
            continue

        node_Add = map_input_Add[node_reduce.output[0]]
        if node_Add.output[0] not in map_input_Sqrt:
            continue

        node_Sqrt = map_input_Sqrt[node_Add.output[0]]
        if node_Sqrt.output[0] not in map_input_Div:
            continue
 
        node_Div = map_input_Div[node_Sqrt.output[0]]
        if node_Div.output[0] not in map_input_Mul:
            continue

        node_Mul = map_input_Mul[node_Div.output[0]]

        if node_Mul.input[0] != node_Div.output[0]:
            layer_norm_input.append(node_Mul.input[0])
        else:
            layer_norm_input.append(node_Mul.input[1])

        if node_Mul.output[0] not in map_input_Add:
            continue

        node_Add1 = map_input_Add[node_Mul.output[0]]
        layer_norm_input.append(node_Add1.input[1])

        removed_nodes.append(node)
        removed_nodes.append(node_sub)
        removed_nodes.append(node_pow)
        removed_nodes.append(node_reduce)
        removed_nodes.append(node_Add)
        removed_nodes.append(node_Sqrt)
        removed_nodes.append(node_Div)
        removed_nodes.append(node_Mul)
        removed_nodes.append(node_Add1)
        removed_nodes.append(map_output_Constant[node_pow.input[1]])

        removed_nodes.append(map_output_Constant[node_Add.input[1]])
        layer_norm_output.append(node_Add1.output[0])
        id = id + 1
        layer_norm_output.append('saved_mean_' + str(id))
        id = id + 1
        layer_norm_output.append('saved_inv_std_' + str(id))
        layer_norm = onnx.helper.make_node("LayerNormalization",
                                        layer_norm_input,
                                        layer_norm_output,
                                        "LayerNormalization_" + str(id),
                                        None,
                                        axis = node_reduce.attribute[0].ints[0],
                                        epsilon = 9.999999960041972e-13)
        layer_norm_nodes.append(layer_norm)

    # remove left side Subs
    for node in graph_proto.node:
        if node.op_type == 'Sub':
            if node.input[1] in first_ReduceMean_outputs:
                removed_nodes.append(node)

    all_nodes = []
    for node in graph_proto.node:
        if node not in removed_nodes:
            all_nodes.append(node)

    for node in layer_norm_nodes:
        all_nodes.append(node)

    graph_proto.ClearField("node")
    graph_proto.node.extend(all_nodes)
