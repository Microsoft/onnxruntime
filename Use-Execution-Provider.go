import onnxruntime as rt

#define the priority order for the execution providers
# prefer CUDA Execution Provider over CPU Execution Provider
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# initialize the model.onnx
sess = rt.InferenceSession("model.onnx", providers=EP_list)

# get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
output_name = sess.get_outputs()[0].name

# get the inputs metadata as a list of :class:`onnxruntime.NodeArg`
input_name = sess.get_inputs()[0].name

# inference run using image_data as the input to the model 
detections = sess.run([output_name], {input_name: image_data})[0]

print("Output shape:", detections.shape)

# Process the image to mark the inference points 
image = post.image_postprocess(original_image, input_size, detections)
image = Image.fromarray(image)
image.save("kite-with-objects.jpg")

# Update EP priority to only CPUExecutionProvider
sess.set_providers('CPUExecutionProvider')

cpu_detection = sess.run(...)
