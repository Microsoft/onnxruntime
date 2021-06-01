---
title: Quantize ONNX Models
parent: How to
nav_order: 4
---
# Quantize ONNX Models
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Quantization Overview
Quantization in ONNX Runtime refers to 8 bit linear quantization of an ONNX model.

 During quantization the floating point real values are mapped to an 8 bit quantization space and it is of the form:
 VAL_fp32 = Scale * (VAL_quantized - Zero_point)

 Scale is a positive real number used to map the floating point numbers to a quantization space. It is calculated as follows:

 For unsigned 8 bit
 ```
 scale = (data_range_max - data_range_min) / (quantization_range_max - quantization_range_min)
 ```

 For signed 8 bit
 ```
 scale = Abs(data_range_max, data_range_min) * 2 / (quantization_range_max - quantization_range_min)
 ```

 Zero point represents zero in the quantization space. It is important that the floating point zero value be exactly representable in quantization space. This is because zero padding is used in many CNNs. If it is not possible to represent 0 uniquely after quantization, it will result in accuracy errors.

## Quantizing an ONNX model
There are 3 ways of quantizing a model: dynamic, static and quantize-aware training quantization.

* **Dynamic quantization**: This method calculates the quantization parameter (scale and zero point) for activations dynamically.

* **Static quantization**: It leverages the calibration data to calculates the quantization parameter of activations.

* **Quantize-Aware training quantization**: The quantization parameter of activation are calculated while training, and the training process can control activation to a certain range.

### List of Supported Quantized Ops
The following ops were chosen as phase 1 ops because these ops consume the most compute and power in most CNN models.
 * Conv
 * MatMul
 * MaxPool
 * Relu
 * Clip
 * Add (Experimental)
 * Mul (Experimental)

 ### Quantization and model opset versions
 Quantization ops were introduced in ONNX opset version 10, so the model which is being quantized must be opset 10 or higher. If the model opset version is < 10 then the model should be reconverted to ONNX from its original framework using a later opset.

### Quantization and Graph Optimization
Quantization and graph optimizations may not always work together. The model owner should be aware of this and run perf evaluations to understand which technique provides the best performance for their model.

* **Quantizing an optimized model**

    If a model is optimized using level 99 (i.e. all possible optimizations are run on that model) then it is possible that quantization cannot be applied on this model anymore. In this case, running the quantization script will not affect the model.

* **Optimizing a quantized model**

    The same holds the other way around. After quantizing a model, some graph optimizations which otherwise might have been applicable may not be applicable anymore.


## Quantization API
Quantization has 3 main APIs:
* quantize_dynamic: dynamic quantization
* quantize_static: static quantization
* quantize_qat: quantize-aware training quantization

### Options
{: .no_toc }
See below for a description of the common options to quantize_dynamic, quantize_static and quantize_qat:

- **model_input**:

    file path of model to quantize
- **model_output**:

    file path of model to quantize
- **op_types_to_quantize**: *default: []*

    specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default.
- **per_channel**: *default: False*

    If True, weights of Conv nodes are quantized per output channel.
  
    If False, they are quantized per tensor. Refer [QLinearConv](https://github.com/onnx/onnx/blob/master/docs/Operators.md#qlinearconv) for more information.
- **activation_type**: *default: QuantType.QUInt8*

    quantization data type of activation. It can be QuantType.QInt8 or QuantType.QUInt8
- **weight_type**: *default: QuantType.QUInt8*

    quantization data type of weight. It can be QuantType.QInt8 or QuantType.QUInt8
- **nodes_to_quantize**: *default: []*

    List of nodes names to quantize. When this list is not None only the nodes in this list
    are quantized.
    example:
    [
        'Conv__224',
        'Conv__252'
    ]
- **nodes_to_exclude**: *default: []*

    List of nodes names to exclude. The nodes in this list will be excluded from quantization
    when it is not None.

In addition, the user needs to provide an implementation of CalibrationDataReader for quantize_static CalibrationDataReader to take in the calibration data and generate the input

### Example
- Dynamic quantization
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'path/to/the/model.onnx'
model_quant = 'path/to/the/model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
```

- QAT quantization
```python
import onnx
from onnxruntime.quantization import quantize_qat, QuantType

model_fp32 = 'path/to/the/model.onnx'
model_quant = 'path/to/the/model.quant.onnx'
quantized_model = quantize_qat(model_fp32, model_quant)
```

- Static quantization

  Please refer to [E2E_example_model](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/quantization/E2E_example_model) for an example of static quantization.

### Calibration support for Static Quantization
#### MinMax static calibration
This Quantization tool also provides API for generating calibration table using MinMax algorithm, as previously mentioned, users need to provide implementation of CalibrationDataReader.```data_reader.py``` is an example of data reader implementaion with both serial and batch processing.
After calling the API, three different format of calibration tables are generated with filename calibration.* (FlatBuffers, Python dictionary and plain text).
Note: In order to include all tensors from the model for better calibration, please run symbolic_shape_infer.py first. (see [here](./../reference/execution-providers/TensorRT-ExecutionProvider.html#sample))
#### Example
{: .no_toc }
```
data_reader = YoloV3DataReader(calibration_dataset, model_path=augmented_model_path)
calibrate.collect_data(data_reader)
calibrate.compute_range()
```
Please see [E2E_example_model/e2e_user_yolov3_example.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/E2E_example_model/object_detection/trt/yolov3/e2e_user_yolov3_example.py) for more details.
### Evaluation for quantization
#### COCO dataset evaluation
This tool integrates the COCO dataset API to evaluate model prediction. Please make sure to install COCO API first (```pip install pycocotools```).
#### Example
{: .no_toc }
```
dr = YoloV3DataReader(validation_dataset, model_path=model_path, start_index=i, size_limit=stride, batch_size=20, is_evaluation=True)
evaluator = YoloV3Evaluator(model_path, dr, providers=providers)
evaluator.predict()
results += evaluator.get_result()
...
evaluator.evaluate(results, annotations)
```
Please see [E2E_example_model/e2e_user_yolov3_example.py](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/E2E_example_model/object_detection/trt/yolov3/e2e_user_yolov3_example.py) for more details.

