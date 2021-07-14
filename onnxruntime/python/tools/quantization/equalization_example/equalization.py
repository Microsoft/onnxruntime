import onnx
import onnx.numpy_helper
import onnxruntime
import numpy as np
import collections

def update_parameter_names(model, first_input_name, final_output_name):
    """
        By default, the given model may not have a consistent parameter naming convention.
        To resolve this, we rename each node to be a conjunction of its op type and index.
        Then, each parameter of the node is given a name as an extension of the node name.
        The final model is identical topologically, with updated parameter names.
    
        Parameters:
            model (onnx.onnx_ml_pb2.ModelProto): model with parameter names to be updated.
            first_input_name (str): name of the first input to the model.
            final_output_name (str): name of the final output from the model.
        
        Returns:
            model (onnx.onnx_ml_pb2.ModelProto): model with updated parameter names.
    """
    
    name_dict = {first_input_name:first_input_name}

    for i, node in enumerate(model.graph.node):
        node_type = node.op_type

        if node_type == 'Conv':
            old_name_1 = model.graph.node[i].input[1]
            new_name_1 = model.graph.node[i].name + "." + "W"
            name_dict[old_name_1] = new_name_1

            if len(model.graph.node[i].input) == 3:
                old_name_2 = model.graph.node[i].input[2]
                new_name_2 = model.graph.node[i].name + "." + "B"
                name_dict[old_name_2] = new_name_2

            old_out_name = model.graph.node[i].output[0]
            new_out_name = model.graph.node[i].name + "." + "Y"
            name_dict[old_out_name] = new_out_name

        elif node_type == 'BatchNormalization':
            old_name_1 = model.graph.node[i].input[1]
            new_name_1 = model.graph.node[i].name + "." + "scale"
            name_dict[old_name_1] = new_name_1

            old_name_2 = model.graph.node[i].input[2]
            new_name_2 = model.graph.node[i].name + "." + "B"
            name_dict[old_name_2] = new_name_2

            old_name_3 = model.graph.node[i].input[3]
            new_name_3 = model.graph.node[i].name + "." + "mean"
            name_dict[old_name_3] = new_name_3

            old_name_4 = model.graph.node[i].input[4]
            new_name_4 = model.graph.node[i].name + "." + "var"
            name_dict[old_name_4] = new_name_4

            old_out_name = model.graph.node[i].output[0]
            new_out_name = model.graph.node[i].name + "." + "Y"
            name_dict[old_out_name] = new_out_name

        elif node_type =='Relu':

            old_out_name = model.graph.node[i].output[0]
            new_out_name = model.graph.node[i].name + "." + "Y"
            name_dict[old_out_name] = new_out_name

        elif node_type == 'Add':

            old_out_name = model.graph.node[i].output[0]
            new_out_name = model.graph.node[i].name + "." + "Y"
            name_dict[old_out_name] = new_out_name

        elif node_type == 'ReduceMean':

            old_out_name = model.graph.node[i].output[0]
            new_out_name = model.graph.node[i].name + "." + "Y"
            name_dict[old_out_name] = new_out_name

        elif node_type == 'Gemm':
            old_name_1 = model.graph.node[i].input[1]
            new_name_1 = model.graph.node[i].name + "." + "B"
            name_dict[old_name_1] = new_name_1

            old_name_2 = model.graph.node[i].input[2]
            new_name_2 = model.graph.node[i].name + "." + "C"
            name_dict[old_name_2] = new_name_2

            old_out_name = model.graph.node[i].output[0]
            new_out_name = model.graph.node[i].name + "." + "Y"
            name_dict[old_out_name] = new_out_name

        else:
            print("Unmapped node:", node_type)
            raise NotImplementedError

    name_dict[final_output_name] = final_output_name
    
    for i, node in enumerate(model.graph.node):
    
        new_inputs = []

        for j, param in reversed(list(enumerate(node.input))):

            old_name = model.graph.node[i].input[j]
            new_name = name_dict[old_name]
            new_inputs.append(new_name)
            model.graph.node[i].input.pop()

        for j, param in reversed(list(enumerate(new_inputs))):

            model.graph.node[i].input.append(param)

        old_output = model.graph.node[i].output[0]
        new_output = name_dict[old_output]
        model.graph.node[i].output.pop()
        model.graph.node[i].output.append(new_output)
        
    for i, init in enumerate(model.graph.initializer):
        old_init_name = model.graph.initializer[i].name
        new_init_name = name_dict[old_init_name]

        model.graph.initializer[i].name = new_init_name
        
    return model

def get_init_lookup(model):
    """
    Create lookup table with each node index and corresponding node name.
    
        Parameters:
            model (onnx.onnx_ml_pb2.ModelProto): model with initializer.
        
        Output:
            init_lookup (dict): mapping from initializer names to respective indecies.
    """
        
    init_lookup = {}
    
    for init_index, init in enumerate(model.graph.initializer):
        
        init_lookup[init.name] = init_index
        
    return init_lookup

def add_zeroed_bias(model, init_lookup):
    """
    By default, some models do not have a bias attribute in their convolution op. When
    merging batchnorm, a bias parameter in the conv layer is necessary as a destination
    for the reassignment of batchnorm parameters. self.model is modified in-place.
    
        Parameters:
            model (onnx.onnx_ml_pb2.ModelProto): model with some conv layers without bias.
            init_lookup (dict): mapping of model initializer names to indecies.
            
        Outputs:
            model (onnx.onnx_ml_pb2.ModelProto): model with all conv layers with bias.
            init_lookup (dict): updated mapping of model initializer names to indecies.
    """

    for i, node in enumerate(model.graph.node):
        
        node_op_type = model.graph.node[i].op_type
        node_input_len = len(model.graph.node[i].input)

        if node_op_type == "Conv" and node_input_len == 2:
            node_weight_name = model.graph.node[i].name + "." + "W"
            node_bias_name = model.graph.node[i].name + "." + "B"
            
            init_weight_index = init_lookup[node_weight_name]
            node_weight_shape = onnx.numpy_helper.to_array(
                model.graph.initializer[init_weight_index]
            ).shape

            node_bias_shape = node_weight_shape[0]
            node_bias_np = np.zeros(node_bias_shape, dtype=np.float32)
            node_bias_pb = onnx.numpy_helper.from_array(
                node_bias_np, name=node_bias_name
            )

            model.graph.node[i].input.append(node_bias_name)
            model.graph.initializer.append(node_bias_pb)
            
    init_lookup = get_init_lookup(model)

    return model, init_lookup

def merge_batchnorm(model, init_lookup):
    """
    Integrate batchnorm gamma, beta, running mean, and running var into the parametric
    values of the preceeding convolutional layer. Also known as merge batchnorm, this
    operation improves accuracy calculation and reduced latency (if batchnorm is later
    removed). Results in self.model conv layers changed in-place and batchnorm distribution
    being 0 centered with 1 variance.
    
        Parameters:
            model (onnx.onnx_ml_pb2.ModelProto): model with batch norm layers.
            init_lookup (dict): mapping of model initializer names to indecies.
            
        Outputs:
            model (onnx.onnx_ml_pb2.ModelProto): model with batch norm integrated into conv parameters.
            init_lookup (dict): updated mapping of model initializer names to indecies.
    """
    
    for i in range(len(model.graph.node) - 1):
        
        node_1_type = model.graph.node[i].op_type
        node_2_type = model.graph.node[i+1].op_type
        
        node_1_conv_flag = node_1_type == "Conv"
        node_2_bn_flag   = node_2_type == "BatchNormalization"
        
        if node_1_conv_flag and node_2_bn_flag:
            eps = model.graph.node[i+1].attribute[0].f
            
            c_w_name = model.graph.node[i].name + "." + "W"
            c_b_name = model.graph.node[i].name + "." + "B"
            b_w_name = model.graph.node[i+1].name + "." + "scale"
            b_b_name = model.graph.node[i+1].name + "." + "B"
            b_m_name = model.graph.node[i+1].name + "." + "mean"
            b_v_name = model.graph.node[i+1].name + "." + "var"
            
            c_w_index = init_lookup[c_w_name]
            c_b_index = init_lookup[c_b_name]
            b_w_index = init_lookup[b_w_name]
            b_b_index = init_lookup[b_b_name]
            b_m_index = init_lookup[b_m_name]
            b_v_index = init_lookup[b_v_name]
            
            c_w = model.graph.initializer[c_w_index]
            c_b = model.graph.initializer[c_b_index]
            b_w = model.graph.initializer[b_w_index]
            b_b = model.graph.initializer[b_b_index]
            b_m = model.graph.initializer[b_m_index]
            b_v = model.graph.initializer[b_v_index]
    
            c_w = onnx.numpy_helper.to_array(c_w)
            c_b = onnx.numpy_helper.to_array(c_b)
            b_w = onnx.numpy_helper.to_array(b_w)
            b_b = onnx.numpy_helper.to_array(b_b)
            b_m = onnx.numpy_helper.to_array(b_m)
            b_v = onnx.numpy_helper.to_array(b_v)
            
            adj_c_b = c_b * (b_w / np.sqrt(b_v + eps))
            adj_b_b = (b_b - (b_w * b_m) / np.sqrt(b_v + eps))
            new_c_b = adj_c_b + adj_b_b

            new_c_w = (c_w.T * (b_w / np.sqrt(b_v + eps))).T

            new_b_w = np.ones(b_w.shape, dtype=np.float32)
            new_b_b = np.zeros(b_b.shape, dtype=np.float32)
            new_b_m = np.zeros(b_m.shape, dtype=np.float32)
            new_b_v = np.ones(b_v.shape, dtype=np.float32)

            model.graph.initializer[c_w_index].raw_data = onnx.numpy_helper.from_array(new_c_w).raw_data
            model.graph.initializer[c_b_index].raw_data = onnx.numpy_helper.from_array(new_c_b).raw_data
            model.graph.initializer[b_w_index].raw_data = onnx.numpy_helper.from_array(new_b_w).raw_data
            model.graph.initializer[b_b_index].raw_data = onnx.numpy_helper.from_array(new_b_b).raw_data
            model.graph.initializer[b_m_index].raw_data = onnx.numpy_helper.from_array(new_b_m).raw_data
            model.graph.initializer[b_v_index].raw_data = onnx.numpy_helper.from_array(new_b_v).raw_data

            model.graph.node[i+1].attribute[0].f = 0

            i = i + 1
            
    init_lookup = get_init_lookup(model)
    
    return model, init_lookup

def scale(a, s, axis):
    """
    Scale array a by scaling factor s. s is applied to each
    sub-array along specified axis.
    # https://stackoverflow.com/a/30032182/5196692
    
        Parameters:
            a (numpy.ndarray): numpy array of model weights.
            s (numpy.ndarray): numpy array of scalings to be applied.
            axis (int): axis to apply scalings.
            
        Outputs:
            mult_out (numpy.ndarray): numpy array with scalings applied.
    """
    
    if len(a.shape) == 1:
        axis = 0
    elif a.shape[1] == 1:
        axis = 0

    given_axis = axis

    dim_array = np.ones((1,a.ndim),int).ravel()
    dim_array[given_axis] = -1

    s_reshaped = s.reshape(dim_array)

    mult_out = a*s_reshaped

    assert(a.shape == mult_out.shape)

    return mult_out

def layer_equalization(W1, W2, b1, bn_W1=None, bn_b1=None, eps=0):
    """
    Calculates the layer equalized scaling parameters of two convolutional layers
    and returns the scaled weights accross their appropriate axis. If signed,
    the full range is used, else the range w.r.t. zero is used.
    
        Parameters:
            W1 (numpy.ndarray): weights of the first conv layer.
            W2 (numpy.ndarray): weights of the second conv layer.
            b1 (numpy.ndarray): bias of the first conv layer.
            bn_W1 (numpy.ndarray): batch norm weights of the first conv layer.
            bn_b1 (numpy.ndarray): batch norm bias of the first conv layer.
            eps (float): batch norm epsilon.
            
        Outputs:
            new_W1 (numpy.ndarray): equalized weights of the first conv layer.
            new_W2 (numpy.ndarray): equalized weights of the second conv layer.
            new_b1 (numpy.ndarray): equalized bias of the first conv layer.
            new_bn_W1 (numpy.ndarray): equalized batch norm weights of the first conv layer.
            new_bn_b1 (numpy.ndarray): equalized batch norm bias of the first conv layer.
            S (numpy.ndarray): scalings used for equalization.
    """
    
    W1_depthwise_flag = True if W1.shape[1] == 1 else False
    W2_depthwise_flag = True if W2.shape[1] == 1 else False

    if W1_depthwise_flag == False and W2_depthwise_flag == False:
        max_1 = np.max(W1, axis=(1, 2, 3)).squeeze()
        min_1 = np.min(W1, axis=(1, 2, 3)).squeeze()

        max_2 = np.max(W2, axis=(0, 2, 3)).squeeze()
        min_2 = np.min(W2, axis=(0, 2, 3)).squeeze()

    elif W1_depthwise_flag == True:
        max_1 = np.max(W1, axis=(1, 2, 3)).squeeze()
        min_1 = np.min(W1, axis=(1, 2, 3)).squeeze()

        max_2 = np.max(W2, axis=(0, 2, 3)).squeeze()
        min_2 = np.min(W2, axis=(0, 2, 3)).squeeze()

    elif W2_depthwise_flag == True:
        max_1 = np.max(W1, axis=(1, 2, 3)).squeeze()
        min_1 = np.min(W1, axis=(1, 2, 3)).squeeze()

        max_2 = np.max(W2, axis=(1, 2, 3)).squeeze()
        min_2 = np.min(W2, axis=(1, 2, 3)).squeeze()

    else:
        raise ValueError

    range_1 = max_1 - min_1
    range_2 = max_2 - min_2

    S = range_1**-1 * np.sqrt(range_1 * range_2 + eps)

    new_W2 = scale(W2, S**-1, axis=1) if W2_depthwise_flag == False else scale(W2, S**-1, axis=0)
    new_W1 = scale(W1, S, axis=0)
    new_b1 = scale(b1, S, axis=0) if b1 is not None else None

    new_bn_W1 = bn_W1
    new_bn_b1 = bn_b1

    return new_W1, new_W2, new_b1, new_bn_W1, new_bn_b1, S

def get_residual_node_list(model):
    """
    Set node identifiers where residual connections occur. Identification is necessary since
    you can equalize to these node but not from these nodes. In other words, equalizing from 
    residual nodes would be a mistake because equalization is for one connecting pair but 
    there are two connecting pairs.
    
        Parameters:
            model (onnx.onnx_ml_pb2.ModelProto): model with residual connections.
            
        Outputs:
            pre_residual_node_names (list): list of node names in model with residual connections.
    """
        
    input_node_names = []
    
    for node in model.graph.node:
        
        input_node_names.append(node.input)

    input_node_names = [item for sublist in input_node_names for item in sublist]
    
    residual_node_names = [x for x, y in collections.Counter(input_node_names).items() if y > 1]
    
    pre_residual_node_names = [node.input[0] for node in model.graph.node if node.output[0] in residual_node_names]
    
    return pre_residual_node_names

def cross_layer_equalization(model, init_lookup, pre_residual_node_names):
    """
    Cross layer equalization is layer equalization applied iteratively
    over conv layer pairs (groups of ops) in the network. There are
    three condition accounted for so far in this implementation:
    1. If the layer pair contains only conv nodes
    2. If the layer pair contains conv nodes and relu activations,
    3. If the first of two conv nodes is followed by batchnorm and both
       have activations.
    Because of the interdependence of the equalization algorithm, the
    conditions relevant to the model must be defined and it cannot simply
    be done in place (as is the case with default quantization). 
    self.model optimizations are applied in-place. 
    
    Parameters:
        model (onnx.onnx_ml_pb2.ModelProto): model to be equalized.
        init_lookup (dict): mapping of model initializer names to indecies.
        pre_residual_node_names (list): list of node names in model with residual connections.
            
        Outputs:
            model (onnx.onnx_ml_pb2.ModelProto): equalized model.
    """

    for i in range(len(model.graph.node)-3):

        node_1_type = model.graph.node[i].op_type
        node_2_type = model.graph.node[i+1].op_type
        node_3_type = model.graph.node[i+2].op_type
        node_4_type = model.graph.node[i+3].op_type

        node_1_conv_flag = node_1_type == "Conv"
        node_2_conv_flag = node_2_type == "Conv"
        node_2_bn_flag   = node_2_type == "BatchNormalization"
        node_3_conv_flag = node_3_type == "Conv"
        node_3_relu_flag = node_3_type == "Relu"
        node_4_conv_flag = node_4_type == "Conv"

        if model.graph.node[i].output[0] in pre_residual_node_names:
            
            continue

        if node_1_conv_flag and node_2_conv_flag:

            c1_w_name = model.graph.node[i] + "." + "W"
            c1_b_name = model.graph.node[i] + "." + "B"
            c2_w_name = model.graph.node[i+1] + "." + "W"
            
            c1_w_index = init_lookup[c1_w_name]
            c1_b_index = init_lookup[c1_b_name]
            c2_w_index = init_lookup[c2_w_name]

            c1_w = model.graph.initializer[c1_w_index]
            c1_b = model.graph.initializer[c1_b_index]
            c2_w = model.graph.initializer[c2_w_index]

            c1_w = onnx.numpy_helper.to_array(c1_w)
            c1_b = onnx.numpy_helper.to_array(c1_b)
            c2_w = onnx.numpy_helper.to_array(c2_w)

            new_c1_w, new_c2_w, new_c1_b, _, _, S = layer_equalization(c1_w, c2_w, c1_b)

            new_c1_w = onnx.numpy_helper.from_array(new_c1_w).raw_data
            new_c1_b = onnx.numpy_helper.from_array(new_c1_b).raw_data
            new_c2_w = onnx.numpy_helper.from_array(new_c2_w).raw_data

            model.graph.initializer[c1_w_index].raw_data = new_c1_w
            model.graph.initializer[c1_b_index].raw_data = new_c1_b
            model.graph.initializer[c2_w_index].raw_data = new_c2_w

        elif node_1_conv_flag and node_2_bn_flag and node_3_conv_flag: 

            c1_w_name = model.graph.node[i].name + "." + "W"
            c1_b_name = model.graph.node[i].name + "." + "B"
            b1_w_name = model.graph.node[i+1].name + "." + "scale"
            b1_b_name = model.graph.node[i+1].name + "." + "B"
            c2_w_name = model.graph.node[i+2].name + "." + "W"
            
            c1_w_index = init_lookup[c1_w_name]
            c1_b_index = init_lookup[c1_b_name]
            b1_w_index = init_lookup[b1_w_name]
            b1_b_index = init_lookup[b1_b_name]
            c2_w_index = init_lookup[c2_w_name]

            c1_w = model.graph.initializer[c1_w_index]
            c1_b = model.graph.initializer[c1_b_index]
            b1_w = model.graph.initializer[b1_w_index]
            b1_b = model.graph.initializer[b1_b_index]
            c2_w = model.graph.initializer[c2_w_index]

            c1_w = onnx.numpy_helper.to_array(c1_w)
            c1_b = onnx.numpy_helper.to_array(c1_b)
            b1_w = onnx.numpy_helper.to_array(b1_w)
            b1_b = onnx.numpy_helper.to_array(b1_b)
            c2_w = onnx.numpy_helper.to_array(c2_w)

            new_c1_w, new_c2_w, new_c1_b, new_b1_w, new_b1_b, S = layer_equalization(c1_w, c2_w, c1_b, bn_W1=b1_w, bn_b1=b1_b)

            new_c1_w = onnx.numpy_helper.from_array(new_c1_w).raw_data
            new_c1_b = onnx.numpy_helper.from_array(new_c1_b).raw_data
            new_b1_w = onnx.numpy_helper.from_array(new_b1_w).raw_data
            new_b1_b = onnx.numpy_helper.from_array(new_b1_b).raw_data
            new_c2_w = onnx.numpy_helper.from_array(new_c2_w).raw_data

            model.graph.initializer[c1_w_index].raw_data = new_c1_w
            model.graph.initializer[c1_b_index].raw_data = new_c1_b
            model.graph.initializer[b1_w_index].raw_data = new_b1_w
            model.graph.initializer[b1_b_index].raw_data = new_b1_b
            model.graph.initializer[c2_w_index].raw_data = new_c2_w

        elif node_1_conv_flag and node_2_bn_flag and node_3_relu_flag and node_4_conv_flag: 

            c1_w_name = model.graph.node[i].name + "." + "W"
            c1_b_name = model.graph.node[i].name + "." + "B"
            b1_w_name = model.graph.node[i+1].name + "." + "scale"
            b1_b_name = model.graph.node[i+1].name + "." + "B"
            c2_w_name = model.graph.node[i+3].name + "." + "W"
            
            c1_w_index = init_lookup[c1_w_name]
            c1_b_index = init_lookup[c1_b_name]
            b1_w_index = init_lookup[b1_w_name]
            b1_b_index = init_lookup[b1_b_name]
            c2_w_index = init_lookup[c2_w_name]

            c1_w = model.graph.initializer[c1_w_index]
            c1_b = model.graph.initializer[c1_b_index]
            b1_w = model.graph.initializer[b1_w_index]
            b1_b = model.graph.initializer[b1_b_index]
            c2_w = model.graph.initializer[c2_w_index]

            c1_w = onnx.numpy_helper.to_array(c1_w)
            c1_b = onnx.numpy_helper.to_array(c1_b)
            b1_w = onnx.numpy_helper.to_array(b1_w)
            b1_b = onnx.numpy_helper.to_array(b1_b)
            c2_w = onnx.numpy_helper.to_array(c2_w)

            new_c1_w, new_c2_w, new_c1_b, new_b1_w, new_b1_b, S = layer_equalization(c1_w, c2_w, c1_b, bn_W1=b1_w, bn_b1=b1_b)

            new_c1_w = onnx.numpy_helper.from_array(new_c1_w).raw_data
            new_c1_b = onnx.numpy_helper.from_array(new_c1_b).raw_data
            new_b1_w = onnx.numpy_helper.from_array(new_b1_w).raw_data
            new_b1_b = onnx.numpy_helper.from_array(new_b1_b).raw_data
            new_c2_w = onnx.numpy_helper.from_array(new_c2_w).raw_data

            model.graph.initializer[c1_w_index].raw_data = new_c1_w
            model.graph.initializer[c1_b_index].raw_data = new_c1_b
            model.graph.initializer[b1_w_index].raw_data = new_b1_w
            model.graph.initializer[b1_b_index].raw_data = new_b1_b
            model.graph.initializer[c2_w_index].raw_data = new_c2_w

        else:
            continue
          
    return model


def equalize(model_path, save=False, save_path="equalized_model.onnx",
             first_input_name = "input", final_output_name = "output"):
    """
    Execute cross layer equalization after adding bias attribute to conv,
    replacing relu6 (clip) with relu, and applying merge batchnorm. Model
    is modified in-place with optimizations to improve quantization.
        
        Parameters:
            model_path (str): path to onnx model.
            save (bool): flag identifying if model should be saved.
            save_path (str): path to equalized onnx model.
            first_input_name (str): name of the first input to the model.
            final_output_name (str): name of the final output from the model.
            
        Outputs:
            model (onnx.onnx_ml_pb2.ModelProto): equalized onnx model.
    """
    
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    model = update_parameter_names(model, first_input_name, final_output_name)
    onnx.checker.check_model(model)

    init_lookup = get_init_lookup(model)
    model, init_lookup = add_zeroed_bias(model, init_lookup)
    onnx.checker.check_model(model)

    model, init_lookup = merge_batchnorm(model, init_lookup)
    onnx.checker.check_model(model)

    pre_residual_node_names = get_residual_node_list(model)
    model = cross_layer_equalization(model, init_lookup, pre_residual_node_names)
    onnx.checker.check_model(model)

    if save==True:
        onnx.save(model, save_path)
        
    return model