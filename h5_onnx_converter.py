import tensorflow as tf
import tf2onnx
import onnx

# Load the model
model = tf.keras.models.load_model('tf_pruned_model.h5')

# Convert TensorFlow model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=14, output_path="tf_pruned_model.onnx")

# Load the ONNX model
onnx_model = onnx.load("tf_pruned_model.onnx")

# Check the model
onnx.checker.check_model(onnx_model)
print("The model is valid: ", onnx.helper.printable_graph(onnx_model.graph))


