---
nav_exclude: true
---
# Create and run PyTorch models with custom operators

ONNX Runtime custom operators can be used to export and run a PyTorch model, when those operators
are not already present in the set of standard ONNX operators.

ONNX Runtime supplies a library of commonly used custom operators via the [onnxruntime_extensions](https://github.com/microsoft/onnxruntime-extensions) package.

You can also write your own custom operators.

This tutorial demonstrates how to utilize custom operators to create and run a PyTorch model.

## Create a PyTorch model with an operator from the extensions library

This example uses the Inverse operator, which is not present in ONNX, but is available
in the custom ops library.

There are three steps to creating an ONNX model that uses an operator from the
ONNX Runtime Extension library.

1. Create or reference a model that uses an operator from the extension library
2. Register this operator (and any others that you use) with the PyTorch ONNX exporter
   Note: The domain name should always be fixed to `ai.onnx.contrib`
3. Export the model using the PyTorch exporter, supplying the model, sample input so that the model
   can be traced into an ONNX computation graph, and a file to write it to. Specifying `verbose=True`
   will print out the graph once it is exported. This is optional, but may be useful as
   a visual verification that your model has been exported as you expected.

```python
import torch
import onnx
import onnxruntime

# Define model that uses an operator that is present in the extensions library
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return torch.inverse(x) + x

model = MyModel()

# Train the model
# In a real use case, you would train your model here

# Register the signature of the operator with the PyTorch exporter
def my_inverse(g, self):
    return g.op("ai.onnx.contrib::Inverse", self)

torch.onnx.register_custom_op_symbolic('::inverse', my_inverse, 1)

# Export the model
input = torch.randn(3, 3)
torch.onnx.export(model, (input, ), "mymodel.onnx", verbose=True)
```

## Run a model with extension operators using the Python inference API

Once the model has been created and exported, you can use the ONNX Runtime inference
APIs to run the model. In order to use the operators defined in the extensions library,
you need to import the extensions library and register it using the InferenceSession
session options.

```python
import onnxruntime
import onnxruntime_extensions
import torch

so = onnxruntime.SessionOptions()
so.register_custom_ops_library(onnxruntime_extensions.get_library_path())

sess = onnxruntime.InferenceSession("mymodel.onnx", so)
input_name = sess.get_inputs()[0].name

# Run with the identity matrix
output = sess.run(None, {input_name: torch.eye(3).numpy()})[0]
print(output)
```

You should see output similar to the following, as the inverse of the identity matrix
is the identity itself and you add it to itself.

```bash
tensor([[2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.]])
```

Operators in the ONNX Runtime Extensions library are listed [here](https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_text_ops.md).

## Create a PyTorch model with a custom operator that you define

If the operator is not in the ONNX Runtime Extensions library, you can write your own
custom operator, as part of your PyTorch model, and export it to ONNX format.

In the model below, the operator is `trace`, which calculates the sum of the
diagonal elements of a matrix.

```python
import torch
import onnx
import onnxruntime

# Define model that uses an operator that is not present in the extensions library
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return torch.trace(x)

model = MyModel()

# Train the model
# ...

# Register the signature with the PyTorch exporter
def my_trace(g, self):
    return g.op("ai.onnx.contrib::Trace", self)

torch.onnx.register_custom_op_symbolic('::trace', my_trace, 1)

# Export the model
input = torch.randn(3, 3)
torch.onnx.export(model, (input, ), "model_with_trace.onnx", verbose=True)
```

## Run a model with custom operators using the Python inference API

Again, once the model has been created and exported, you can use the ONNX Runtime inference
APIs to run the model. Two steps are required before using the ONNX Runtime inference APIs.
Note that these two steps must be executed in this order.

1. Provide a definition for the operator.
   This is achieved by the `@onnx_op` annotation, which adds your operator to the
   library of custom operators already in the extensions library.
2. Register the operator definition with ONNX Runtime
   The previous step added the operator to the extensions library. This step
   makes all operators in the extensions library, as well as your new
   operator available to ONNX Runtime.

```python
import numpy
import torch
import onnxruntime
import onnxruntime_extensions
from onnxruntime_extensions import PyOp, onnx_op

# Define the operator
@onnx_op(op_type="Trace", inputs=[PyOp.dt_float])
def trace(x):
    return numpy.trace(x)

# Register the extensions library (of which your operator is now effectively a member)
so = onnxruntime.SessionOptions()
so.register_custom_ops_library(onnxruntime_extensions.get_library_path())

sess = onnxruntime.InferenceSession("model_with_trace.onnx", so)
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: torch.randn(3, 3).numpy()})[0]
print(output)
```

## Run a model with extension operators using the C++ API

1. Install the ONNX Runtime C++ libraries

    ```bash
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.0/onnxruntime-linux-x64-1.8.0.tgz
    tar xvzf onnxruntime-linux-x64-1.8.0.tgz
    sudo cp onnxruntime-linux-x64-1.8.0/include/* /usr/local/include
    sudo cp cp onnxruntime-linux-x64-1.8.0/lib/* /usr/local/lib
    sudo ldconfig
    ```

2. Build and install the extensions library C++ library

    To run the model using the C++ API you need to build the extensions library from source.
    Note that the build requires 8GB of RAM to execute.

    ```bash
    git clone https://github.com/microsoft/onnxruntime-extensions
    cd onnxruntime-extensions
    ./build.sh
    cp out/Linux/libortcustomops.so /usr/local/lib/   
    sudo ldconfig
    ```

3. Write your application

    This C++ code is identical to the basic C++ sample, with the addition of code to
    load the extensions library at the top.

    ```cpp
    #include <assert.h>
    #include <vector>
    #include <onnxruntime_cxx_api.h>

    // main() is where program execution  begins.
    int main()
    {

        Ort::SessionOptions session_options;
        const char *custom_op_library_filename = "/usr/local/lib/libortcustomops.so";
        void *handle = nullptr;

        // The line loads the customop library into ONNXRuntime engine to load the ONNX model with the custom op
        Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions *)session_options, custom_op_library_filename, &handle));

        //*************************************************************************
        // initialize  enviroment...one enviroment per process
        // enviroment maintains thread pools and other state info
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

        // initialize session options if needed
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        const char *model_path = "mymodel.onnx";

        printf("Using ONNX Runtime C++ API\n");
        Ort::Session session(env, model_path, session_options);

        //*************************************************************************
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;

        // print number of model input nodes
        size_t num_input_nodes = session.GetInputCount();
        std::vector<const char *> input_node_names(num_input_nodes);
        std::vector<int64_t> input_node_dims;

        printf("Number of inputs = %zu\n", num_input_nodes);

        // iterate over all input nodes
        for (int i = 0; i < num_input_nodes; i++)
        {
            // print input node names
            char *input_name = session.GetInputName(i, allocator);
            printf("Input %d : name=%s\n", i, input_name);
            input_node_names[i] = input_name;

            // print input node types
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Input %d : type=%d\n", i, type);

            // print input shapes/dims
            input_node_dims = tensor_info.GetShape();
            printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
            for (int j = 0; j < input_node_dims.size(); j++)
                printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
        }

        // Results should be...
        // Number of inputs = 1
        // Input 0 : name = data_0
        // Input 0 : type = 1
        // Input 0 : num_dims = 2
        // Input 0 : dim 0 = 3
        // Input 0 : dim 1 = 3

        //*************************************************************************
        // Similar operations to get output node information.
        // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
        // OrtSessionGetOutputTypeInfo() as shown above.

        //*************************************************************************
        // Run the model using sample data, and inspect values

        size_t input_tensor_size = 3 * 3;

        std::vector<float> input_tensor_values(input_tensor_size);
        std::vector<const char *> output_node_names = {"2"};

        // Run with the identity matrix 
        input_tensor_values[0] = 1;
        input_tensor_values[1] = 0;
        input_tensor_values[2] = 0;
        input_tensor_values[3] = 0;
        input_tensor_values[4] = 1;
        input_tensor_values[5] = 0;
        input_tensor_values[6] = 0;
        input_tensor_values[7] = 0;
        input_tensor_values[8] = 1;

        // Create input tensor object from data values
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 2);
        assert(input_tensor.IsTensor());

        // Run model & input tensor, get back output tensor
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

        // Get pointer to output tensor float values
        float *floatarr = output_tensors.front().GetTensorMutableData<float>();

        // Run the model, and print the output
        for (int i = 0; i < 9; i++)
            printf("Value [%d] =  %f\n", i, floatarr[i]);

        printf("Done!\n");
        return 0;
    ```

4. Build and run your application

    ```bash
    g++ Sample.cpp -lonnxruntime -o sample
    ./sample
    ```

    You should see the following output from the model

    ```bash
    tensor([[2., 0., 0.],
            [0., 2., 0.],
            [0., 0., 2.]])
    ```
