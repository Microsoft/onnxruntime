# Convert And Inference Pytorch model with CustomOps

With [onnxruntime_customops](https://github.com/microsoft/onnxruntime-extensions) package, the PyTorch model with the operation cannot be converted into the standard ONNX operators still be converted and the converted ONNX model still can be run with ONNXRuntime, plus onnxruntime_customops package. This tutorial show it works

## Converting
Suppose there is a model which cannot be converted because there is no matrix inverse operation in ONNX standard opset. And the model will be defined like the following.


```python
import torch
import torchvision

class CustomInverse(torch.nn.Module):
    def forward(self, x):
        return torch.inverse(x) + x
```

To export this model into ONNX format, we need register a custom op handler for pytorch.onn.exporter.


```python
from torch.onnx import register_custom_op_symbolic


def my_inverse(g, self):
    return g.op("ai.onnx.contrib::Inverse", self)

register_custom_op_symbolic('::inverse', my_inverse, 1)
```

Then, invoke the exporter


```python
import io
import onnx

x0 = torch.randn(3, 3)
# Export model to ONNX
f = io.BytesIO()
t_model = CustomInverse()
torch.onnx.export(t_model, (x0, ), f, opset_version=12)
onnx_model = onnx.load(io.BytesIO(f.getvalue()))
```

Now, we got a ONNX model in the memory, and it can be save into a disk file by 'onnx.save_model(onnx_model, <file_path>)

## Inference
This converted model cannot directly run the onnxruntime due to the custom operator. but it can run with onnxruntime_customops easily.

Firstly, let define a PyOp function to inteprete the custom op node in the ONNNX model.


```python
import numpy
from onnxruntime_customops import onnx_op, PyOp
@onnx_op(op_type="Inverse")
def inverse(x):
    # the user custom op implementation here:
    return numpy.linalg.inv(x)

```

* **ONNX Inference**


```python
from onnxruntime_customops import PyOrtFunction
onnx_fn = PyOrtFunction.from_model(onnx_model)
y = onnx_fn(x0.numpy())
print(y)
```

    [[-3.081008    0.20269153  0.42009977]
     [-3.3962293   2.5986686   2.4447646 ]
     [ 0.7805753  -0.20394287 -2.7528977 ]]
    

* **Compare the result with Pytorch**


```python
t_y = t_model(x0)
numpy.testing.assert_almost_equal(t_y, y, decimal=5)
```

## Implement the customop in C++ (optional)
To make the ONNX model with the CustomOp runn on all other language supported by ONNX Runtime and be independdent of Python, a C++ implmentation is needed, check here for the [inverse.hpp](https://github.com/microsoft/onnxruntime-extensions/blob/main/operators/math/inverse.hpp) for an example on how to do that.


```python
from onnxruntime_customops import enable_custom_op
# disable the PyOp function and run with the C++ function
enable_custom_op(False)
y = onnx_fn(x0.numpy())
print(y)
```

    [[-3.081008    0.20269153  0.42009977]
     [-3.3962293   2.5986686   2.4447646 ]
     [ 0.7805753  -0.20394287 -2.7528977 ]]
    
