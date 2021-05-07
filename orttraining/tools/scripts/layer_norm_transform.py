import sys
import os.path
from onnx import *
import onnx
import numpy as np

def find_node(graph_proto, op_type):
    nodes = []
    map_input_node = {}
    for node in graph_proto.node:
        if node.op_type == op_type:
            node_input = node.input[1] if op_type == 'Div' or op_type == 'Mul' else node.input[0]
            nodes.append(node)
            map_input_node[node_input] = node
    return nodes, map_input_node

def gen_attribute(key, value):
    attr = AttributeProto()
    attr.name = key
    attr.ints.extend(int(v) for v in value)
    attr.type = AttributeProto.INTS
    return attr

def main():
    if len(sys.argv) < 2:
        print("Please give model path...")
        return

    model_file_path = sys.argv[1]
    #model_file_path = os.path.dirname(sys.argv[1:])
    print("model_file_path: " + model_file_path)
    model_file_name = os.path.basename(model_file_path)
    print("model_file_name: "+ model_file_name)

    new_model_file_path = model_file_path[:-5] + '_layer_norm.onnx'
    print(new_model_file_path)

    model_proto = onnx.load(model_file_path)
    #print(model_proto)

    graph_proto = model_proto.graph
    #print(graph_proto)
    #print(graph_proto.input)

    nodes_Div,  map_input_Div = find_node(graph_proto, 'Div')
    #print(map_input_Div)
    nodes_Sqrt,  map_input_Sqrt = find_node(graph_proto, 'Sqrt')
    #print(map_input_Sqrt)
    nodes_Add,  map_input_Add = find_node(graph_proto, 'Add')
    #print(map_input_Add)
    nodes_ReduceMean,  map_input_ReduceMean = find_node(graph_proto, 'ReduceMean')
    #print(map_input_ReduceMean)
    nodes_Pow,  map_input_Pow = find_node(graph_proto, 'Pow')
    #print(map_input_Pow)
    nodes_Mul,  map_input_Mul = find_node(graph_proto, 'Mul')

    # find right side Sub
    nodes_Sub = []
    map_input_Sub = {}
    for node in graph_proto.node:
        if node.op_type == 'Sub':
            if node.output[0] in map_input_Pow:
                nodes_Sub.append(node)
                map_input_Sub[node.input[1]] = node
    #print(map_input_Sub)

    # find first ReduceMean
    first_ReduceMean = []
    first_ReduceMean_outputs = []
    for node in nodes_ReduceMean:
        if node.output[0] in map_input_Sub:
            first_ReduceMean.append(node)
            first_ReduceMean_outputs.append(node.output[0])
    #print(first_ReduceMean)

    # find constant node
    nodes_Constant = []
    map_output_Constant = {}
    for node in graph_proto.node:
        if node.op_type == 'Constant':
            nodes_Constant.append(node)
            map_output_Constant[node.output[0]] = node
    #print(map_input_Sub)

    id = 0
    removed_nodes = []
    layer_norm_nodes = []
    # Replace with layer norm
    for node in first_ReduceMean:
        layer_norm_input = []
        layer_norm_output = []
        layer_norm_input.append(node.input[0])
        node_sub = map_input_Sub[node.output[0]]
        node_pow = map_input_Pow[node_sub.output[0]]
        node_reduce = map_input_ReduceMean[node_pow.output[0]]
        node_Add = map_input_Add[node_reduce.output[0]]
        node_Sqrt = map_input_Sqrt[node_Add.output[0]]
        node_Div = map_input_Div[node_Sqrt.output[0]]
        node_Mul = map_input_Mul[node_Div.output[0]]
        layer_norm_input.append(node_Mul.input[0])
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
        #print(map_output_Constant[node_Add.input[1]])
        removed_nodes.append(map_output_Constant[node_Add.input[1]])
        layer_norm_output.append(node_Add1.output[0])
        id = id + 1
        layer_norm_output.append('saved_mean_' + str(id))
        id = id + 1
        layer_norm_output.append('saved_inv_std_' + str(id))
        layer_norm = helper.make_node("LayerNormalization",
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

    with open(new_model_file_path, 'wb') as f:
        f.write(model_proto.SerializeToString())

    # Use ORT to verify the converted model. Notice that you must use python package from the
    # training branch because training requires some extra ops.
    import onnxruntime as ort
    # We convert model to accept variable-length batch size, so it can be any positive integer.
    batch = 3
    # This should match --max_seq_length when calling nv_run_pretraining.py.
    sq_length = 512
    # This should match vocab_size in bert_config.json in DeepLearningExamples/PyTorch/LanguageModeling/BERT.
    vocab_size = 30528

    # Create a fake data point.
    vocab_size = 30528 # It shoudl match the value from BERT config file.
    input_ids = np.random.randint(low=0, high=vocab_size, size=(batch, sq_length), dtype=np.int64)
    segment_ids = np.random.randint(low=0, high=2, size=(batch, sq_length), dtype=np.int64)
    input_mask = np.ones((batch, sq_length), dtype=np.int64)

    # Do forward using the original model.
    sess = ort.InferenceSession(model_file_path)
    result = sess.run(None, {'input1': input_ids, 'input2': segment_ids, 'input3': input_mask})

    # Do forward using the new model.
    new_sess = ort.InferenceSession(new_model_file_path)
    new_result = new_sess.run(None, {'input1': input_ids, 'input2': segment_ids, 'input3': input_mask})

    # Compare the outcomes from the two models.
    print(np.linalg.norm(result[0]-new_result[0]))
    print(np.linalg.norm(result[1]-new_result[1]))

if __name__ == "__main__":
    main()
