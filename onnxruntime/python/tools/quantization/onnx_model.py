import onnx
import itertools
from .quant_utils import find_by_name, attribute_to_kwarg
from pathlib import Path

class ONNXModel:
    def __init__(self, model):
        self.model = model

    def nodes(self):
        return self.model.graph.node

    def initializer(self):
        return self.model.graph.initializer

    def graph(self):
        return self.model.graph

    def ir_version(self):
        return self.model.ir_version

    def opset_import(self):
        return self.model.opset_import

    def remove_node(self, node):
        if node in self.model.graph.node:
            self.model.graph.node.remove(node)

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node):
        self.model.graph.node.extend([node])

    def add_nodes(self, nodes_to_add):
        self.model.graph.node.extend(nodes_to_add)

    def add_initializer(self, tensor):
        if find_by_name(tensor.name, self.model.graph.initializer) is None:
            self.model.graph.initializer.extend([tensor])

    def get_initializer(self, name):
        for tensor in self.model.graph.initializer:
            if tensor.name == name:
                return tensor
        return None

    def remove_initializer(self, tensor):
        if tensor in self.model.graph.initializer:
            self.model.graph.initializer.remove(tensor)
            for input in self.model.graph.input:
                if input.name == tensor.name:
                    self.model.graph.input.remove(input)
                    break

    def remove_initializers(self, init_to_remove):
        for initializer in init_to_remove:
            self.remove_initializer(initializer)

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for node in self.model.graph.node:
            for input_name in node.input:
                if input_name not in input_name_to_nodes:
                    input_name_to_nodes[input_name] = [node]
                else:
                    input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_node(self):
        output_name_to_node = {}
        for node in self.model.graph.node:
            for output_name in node.output:
                output_name_to_node[output_name] = node
        return output_name_to_node

    def get_children(self, node, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = []
        for output in node.output:
            if output in input_name_to_nodes:
                for node in input_name_to_nodes[output]:
                    children.append(node)
        return children

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_parent(self, node, idx, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if len(node.input) <= idx:
            return None

        input = node.input[idx]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]

    def find_node_by_name(self, node_name, new_nodes_list, graph):
        '''
        Find out if a node exists in a graph or a node is in the
        new set of nodes created during quantization. Return the node found.
        '''
        graph_nodes_list = list(graph.node)  #deep copy
        graph_nodes_list.extend(new_nodes_list)
        node = find_by_name(node_name, graph_nodes_list)
        return node

    def find_nodes_by_initializer(self, graph, initializer):
        '''
        Find all nodes with given initializer as an input.
        '''
        nodes = []
        for node in graph.node:
            for node_input in node.input:
                if node_input == initializer.name:
                    nodes.append(node)
        return nodes

    @staticmethod
    def __get_initializer(name, graph_path):
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None

    @staticmethod
    def __replace_gemm_with_matmul(graph_path):
        new_nodes = []
        graph = graph_path[-1]
        for node in graph.node:
            graph_attrs = [attr for attr in node.attribute if attr.type == 5 or attr.type == 10]
            if len(graph_attrs):
                node_name = node.name
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == 5:
                        graph_path.append(attr.g)
                        kv = {attr.name: ONNXModel.__replace_gemm_with_matmul(graph_path)}
                    elif attr.type == 10:
                        value = []
                        for subgraph in attr.graphs:
                            graph_path.append(subgraph)
                            value.extend([ONNXModel.__replace_gemm_with_matmul(graph_path)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(node.op_type, node.input, node.output, name=node.name, **kwargs)

            if node.op_type == 'Gemm':
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == 'alpha':
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'beta':
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transA':
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == 'transB':
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]
                    if transB == 1:
                        B, Bs_graph = ONNXModel.__get_initializer(node.input[1], graph_path)
                        if B:
                            # assume B is not used by any other node
                            B_array = onnx.numpy_helper.to_array(B)
                            B_trans = onnx.numpy_helper.from_array(B_array.T)
                            B_trans.name = B.name
                            Bs_graph.initializer.remove(B)
                            for input in Bs_graph.input:
                                if input.name == inputB:
                                    Bs_graph.input.remove(input)
                                    break
                            Bs_graph.initializer.extend([B_trans])
                        else:
                            inputB += '_Transposed'
                            transpose_node = onnx.helper.make_node('Transpose',
                                                                   inputs=[node.input[1]],
                                                                   outputs=[inputB],
                                                                   name=node.name + '_Transpose' if node.name != "" else "")
                            new_nodes.append(transpose_node)

                    matmul_node = onnx.helper.make_node(
                        'MatMul',
                        inputs=[node.input[0], inputB],
                        outputs=[node.output[0] + ('_MatMul' if len(node.input) > 2 else '')],
                        name=node.name + '_MatMul' if node.name != "" else "")
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx.helper.make_node('Add',
                                                         inputs=[node.output[0] + '_MatMul', node.input[2]],
                                                         outputs=node.output,
                                                         name=node.name + '_Add' if node.name != "" else "")
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        graph.ClearField('node')
        graph.node.extend(new_nodes)
        graph_path.pop()
        return graph

    def replace_gemm_with_matmul(self):
        graph_path = [self.graph()]
        ONNXModel.__replace_gemm_with_matmul(graph_path)

    def save_model_to_file(self, output_path, use_external_data_format=False):
        '''
        Save model to external data, which is needed for model size > 2GB
        '''
        self.topological_sort()
        if use_external_data_format:
            onnx.external_data_helper.convert_model_to_external_data(self.model,
                                                                     all_tensors_to_one_file=True,
                                                                     location=Path(output_path).name + ".data")
        onnx.save_model(self.model, output_path)

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name):
        for node in self.model.graph.node:
            ONNXModel.replace_node_input(node, old_input_name, new_input_name)

    @staticmethod
    def replace_node_output(node, old_output_name, new_output_name):
        assert isinstance(old_output_name, str) and isinstance(new_output_name, str)
        for j in range(len(node.output)):
            if node.output[j] == old_output_name:
                node.output[j] = new_output_name

    def replace_output_of_all_nodes(self, old_output_name, new_output_name):
        for node in self.model.graph.node:
            ONNXModel.replace_node_output(node, old_output_name, new_output_name)

    def remove_unused_constant(self):
        input_name_to_nodes = self.input_name_to_nodes()

        #remove unused constant
        unused_nodes = []
        nodes = self.nodes()
        for node in nodes:
            if node.op_type == "Constant" and not self.is_graph_output(
                    node.output[0]) and node.output[0] not in input_name_to_nodes:
                unused_nodes.append(node)

        self.remove_nodes(unused_nodes)

        ununsed_weights = []
        for w in self.initializer():
            if w.name not in input_name_to_nodes and not self.is_graph_output(w.name):
                ununsed_weights.append(w)
                # Remove from graph.input
                for graph_input in self.graph().input:
                    if graph_input.name == w.name:
                        self.graph().input.remove(graph_input)

        self.remove_initializers(ununsed_weights)

    def is_graph_output(self, output_name):
        for output in self.model.graph.output:
            if output.name == output_name:
                return True
        return False

    # TODO:use OnnxModel.graph_topological_sort(self.model.graph) from transformers.onnx_model
    # Currently it breaks Openvino/Linux training gpu pipeline so hold off for 1.8 release
    def topological_sort(self):
        deps_count = [0]*len(self.nodes()) # dependency count of each node
        deps_to_nodes = {} # input to node indice
        sorted_nodes = []  # initialize sorted_nodes
        for node_idx, node in enumerate(self.nodes()):
            # CANNOT use len(node.input) directly because input can be optional
            deps_count[node_idx] = sum(1 for _ in node.input if _ )
            if deps_count[node_idx] == 0: # Constant doesn't depend on any inputs
                sorted_nodes.append(self.nodes()[node_idx])
                continue

            for input_name in node.input:
                if input_name not in deps_to_nodes:
                    deps_to_nodes[input_name] = [node_idx]
                else:
                    deps_to_nodes[input_name].append(node_idx)

        initializer_names = [init.name for init in self.initializer()]
        graph_input_names = [input.name for input in self.model.graph.input]
        input_names = initializer_names + graph_input_names
        input_names.sort()
        prev_input_name = None
        for input_name in input_names:
            if prev_input_name == input_name:
                continue

            prev_input_name = input_name
            if input_name in deps_to_nodes:
                for node_idx in deps_to_nodes[input_name]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(self.nodes()[node_idx])

        start = 0
        end = len(sorted_nodes)

        while start < end:
            for output in sorted_nodes[start].output:
                if output in deps_to_nodes:
                    for node_idx in deps_to_nodes[output]:
                        deps_count[node_idx] = deps_count[node_idx] - 1
                        if deps_count[node_idx] == 0:
                            sorted_nodes.append(self.nodes()[node_idx])
                            end = end + 1
            start = start + 1

        assert(end == len(self.graph().node)), "Graph is not a DAG"
        self.graph().ClearField('node')
        self.graph().node.extend(sorted_nodes)