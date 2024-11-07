from pathlib import Path
import gc

import numpy as np

import torch
from torch import nn

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from impl.tf.moondream.vision.vision_transformer import VisionEncoder, VitBlock, EncoderWrapper, LinearPatchEmbedding, MLP, VisionProjection


from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image


model_id = "vikhyatk/moondream2"
revision = "2024-05-08"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
# tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


im_path = Path("/home/crulis/Documents/AndroidStudioProjects/AndroidAssistant/app/src/main/assets/Great_White_Shark.jpg")

IMG_SIZE = 378


try:
    if im_path.exists():
        with torch.no_grad():
            im_embd = model.encode_image(Image.open(im_path))

        print(im_embd.shape)
    else:
        print("image not found, skipping")
except Exception as e:
    print(f"error: {e}")
    print("error while reading image, skipping")


tf_model = VisionEncoder()


tf.random.set_seed(42)

dummy_input = tf.random.normal(shape=(1, IMG_SIZE, IMG_SIZE, 3))


out = tf_model(dummy_input)

print(out.shape)


# apply the weights from the pytorch model to the tensorflow model


def transfer_mlp(pytorch_mlp, tf_mlp):
    tf_mlp.fc1.set_weights([pytorch_mlp.fc1.weight.T.detach().numpy(), pytorch_mlp.fc1.bias.detach().numpy()])
    tf_mlp.fc2.set_weights([pytorch_mlp.fc2.weight.T.detach().numpy(), pytorch_mlp.fc2.bias.detach().numpy()])


transfer_mlp(model.vision_encoder.projection.mlp, tf_model.projection.mlp)

tf_model.encoder.model["visual"].patch_embed.set_weights([model.vision_encoder.encoder.model.visual.patch_embed.linear.weight.T.detach().numpy(),
                                                           model.vision_encoder.encoder.model.visual.patch_embed.linear.bias.detach().numpy()])

tf_model.encoder.model["visual"].pos_embed.assign(model.vision_encoder.encoder.model.visual.pos_embed.detach().numpy())

tf_model.encoder.model["visual"].norm.set_weights([model.vision_encoder.encoder.model.visual.norm.weight.detach().numpy(),
                                                model.vision_encoder.encoder.model.visual.norm.bias.detach().numpy()])

tf_model.encoder.model["visual"].norm.epsilon = model.vision_encoder.encoder.model.visual.norm.eps

for i, pt_block in enumerate(model.vision_encoder.encoder.model.visual.blocks):
    tf_block: VitBlock = tf_model.encoder.model["visual"].blocks.layers[i]

    tf_block.attn.proj.set_weights([pt_block.attn.proj.weight.T.detach().numpy(), pt_block.attn.proj.bias.detach().numpy()])
    tf_block.attn.qkv.set_weights([pt_block.attn.qkv.weight.T.detach().numpy(), pt_block.attn.qkv.bias.detach().numpy()])
    tf_block.norm1.set_weights([pt_block.norm1.weight.detach().numpy(), pt_block.norm1.bias.detach().numpy()])
    tf_block.norm2.set_weights([pt_block.norm2.weight.detach().numpy(), pt_block.norm2.bias.detach().numpy()])

    tf_block.norm1.epsilon = pt_block.norm1.eps
    tf_block.norm2.epsilon = pt_block.norm2.eps
    transfer_mlp(pt_block.mlp, tf_block.mlp)


print("weights transferred")


np_dummy = dummy_input.numpy()

with torch.no_grad():
    pt_out = model.vision_encoder(torch.tensor(np_dummy).permute(0, 3, 1, 2))

tf.config.run_functions_eagerly(True)

# check first layer output diff

first_layer_out_pt = model.vision_encoder.encoder.model.visual.patch_embed(torch.tensor(np_dummy).permute(0, 3, 1, 2))
first_layer_out_tf = tf_model.encoder.model["visual"].patch_embed(tf.transpose(dummy_input, perm=[0, 3, 1, 2]))

diff = np.square(first_layer_out_pt.detach().numpy() - first_layer_out_tf.numpy()).sum()

print(f"First layer diff: {diff}")


del model
gc.collect()


tf_out = tf_model(dummy_input)


diff = np.square(pt_out.detach().numpy() - tf_out.numpy()).sum()

print(f"TF model diff: {diff}")


def quantize_layer_predicate(layer):
    if layer.name == model.vision_encoder.encoder.model.visual.patch_embed.linear.name:
        return False
    return True


# Apply selective quantization by modifying the converter's configuration
def modify_quantization(layer):
    if not quantize_layer_predicate(layer):
        # Skip quantization for specific layers
        layer.quantize = False

# convert TF model to TFLite

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [tf.float16]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    # tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    # tf.float16
  ]

tflite_model = converter.convert()

print("TF model converted to TFLite")


# test TFLite model

interpreter = tf.lite.Interpreter(model_content=tflite_model)

print("TFLite model loaded")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np_dummy)

print("invoking interpreter")

interpreter.invoke()

tflite_out = interpreter.get_tensor(output_details[0]['index'])

diff = np.square(pt_out.detach().numpy() - tflite_out).sum()

print(f"TFLite model diff: {diff}")

print("TFLite model tested")


out_tflite_path = Path("moondream_vision_enc.tflite")

print(f"saving model to {out_tflite_path}")

out_tflite_path.write_bytes(tflite_model)

print("model saved")