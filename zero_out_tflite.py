import tensorflow as tf
from tensorflow import lite
import keras

from custom_ops_mod import zero_out


inputs = keras.Input((5, 100))

outputs = zero_out(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.experimental_new_converter = True
# converter.experimental_lower_to_saved_model
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.
    # tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("converting model to TFLite")
tflite_model = converter.convert()
print("done.")

filename = "zero_out.tflite"
with open(filename, "wb") as f:
    print(f"writing tflite model to disk ({filename})")
    f.write(tflite_model)
    f.flush()
print("model written to disk.")

print("creating interpreter")
interpreter = lite.Interpreter(filename)

interpreter.allocate_tensors()
interpreter.invoke()

print("done.")