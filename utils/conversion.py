from functools import partial

import numpy as np

import torch
from torch import nn
import tensorflow as tf
from tensorflow import keras

from larq.layers import QuantDense

from binary.modules import ScaledTernaryLinear

from impl.tf.moondream.ternary import ScaledTernary, ScaledTernaryTf, ScaledTernaryUnpack, ScaledTernaryMatmul, ScaledTernaryPackedMM, ternary_act

from impl.tf.moondream.layers import PhiDecoderLayer
from impl.tf.moondream.model import PhiModel


def tf_tensor_to_torch(tf_tensor):
    return torch.from_numpy(tf_tensor.numpy())


def transfer_dense_params_tf_to_torch(dense_torch: nn.Linear, dense_keras):
    w, b = dense_keras.get_weights()
    w, b = (torch.from_numpy(w), torch.from_numpy(b))
    dense_torch.weight.data = w.T
    dense_torch.bias.data = b
    pass


def transfer_dense_params_torch_to_tf(dense_keras: keras.layers.Layer, dense_torch: nn.Module):
    if isinstance(dense_keras, ScaledTernary):
        assert isinstance(dense_torch, ScaledTernaryLinear)
        w, b, scale = dense_torch.weights, dense_torch.bias, dense_torch.scale
        w = w.detach().numpy()
        b = b.detach().numpy()
        scale = scale.detach().numpy()
        dense_keras.clip_val = dense_torch.maxval

        dense_keras.set_weights([scale, b, w.T])

        pass
    elif isinstance(dense_keras, ScaledTernaryTf):
        # pure Tensorflow implementation
        assert isinstance(dense_torch, ScaledTernaryLinear)
        w, b, scale = dense_torch.weights, dense_torch.bias, dense_torch.scale
        w = tf.convert_to_tensor(w.detach().numpy())
        b = b.detach().numpy()
        scale = scale.detach().numpy()

        ternerized = ternary_act(w)
        packed_weights = dense_keras.pack_fn(tf.reshape(tf.cast(ternerized, dtype=tf.int8), shape=(-1,)))
        packed_weights = packed_weights.numpy()

        dense_keras.clip_val = dense_torch.maxval

        dense_keras.set_weights([scale, b, packed_weights])

        pass
    elif isinstance(dense_keras, (ScaledTernaryMatmul, ScaledTernaryUnpack)):
        assert isinstance(dense_torch, ScaledTernaryLinear)
        w, b, scale = dense_torch.weights, dense_torch.bias, dense_torch.scale
        w = tf.convert_to_tensor(w.detach().numpy())
        b = b.detach().numpy()
        scale = scale.detach().numpy()

        # use the custom packing function provided by our custom LCE
        from larq_compute_engine import ops

        ternerized = ternary_act(w)
        packed_weights = ops.pack_fn(tf.cast(ternerized, dtype=tf.int8))
        packed_weights = packed_weights.numpy()

        dense_keras.clip_val = dense_torch.maxval

        dense_keras.set_weights([scale, b, packed_weights])

        pass
    elif isinstance(dense_keras, ScaledTernaryPackedMM):
        assert isinstance(dense_torch, ScaledTernaryLinear)
        w, b, scale = dense_torch.weights, dense_torch.bias, dense_torch.scale
        w = tf.convert_to_tensor(w.detach().numpy())
        b = b.detach().numpy()
        scale = scale.detach().numpy()

        ternerized = ternary_act(w)
        packed_weights = dense_keras.pack_fn(tf.cast(ternerized, dtype=tf.int8))

        dense_keras.clip_val = dense_torch.maxval

        dense_keras.set_weights([scale, b, packed_weights])

        pass
    else:
        w, b = dense_torch.weight, dense_torch.bias
        w = w.detach().numpy().astype(np.float16)
        b = b.detach().numpy().astype(np.float16)
        
        dense_keras.set_weights([w.T, b])
    pass


def transfer_layernorm_params_tf_to_torch(ln_torch: nn.LayerNorm, ln_keras):
    a, b = ln_keras.get_weights()
    a, b = (torch.from_numpy(a), torch.from_numpy(b))

    ln_torch.epsilon = ln_torch.eps

    ln_torch.weight.data = a
    ln_torch.bias.data = b
    pass


def transfer_layernorm_params_torch_to_tf(ln_keras, ln_torch: nn.LayerNorm):
    a, b = ln_torch.weight, ln_torch.bias
    a = a.detach().numpy()
    b = b.detach().numpy()

    ln_keras.eps = ln_keras.epsilon

    ln_keras.set_weights([a, b])
    pass


def transfer_encoder_block_torch_to_tf(torch_enc: nn.Module, tf_enc: PhiDecoderLayer, past_key_values, atol=1e-4):
    tf.random.set_seed(1234)
    tf_inp = tf.random.normal((1, 5, 2048))
    tf_pids = tf.range(0, 5)[None, :]
    attn_mask = tf.zeros((1, 1, 5, 5))

    tf_enc(tf_inp, attention_mask=attn_mask, position_ids=tf_pids, past_key_value=past_key_values)
    transfer_dense_params_torch_to_tf(tf_enc.mlp.fc1, torch_enc.mlp.fc1)
    transfer_dense_params_torch_to_tf(tf_enc.mlp.fc2, torch_enc.mlp.fc2)
    transfer_dense_params_torch_to_tf(tf_enc.mixer.Wqkv, torch_enc.mixer.Wqkv)
    transfer_dense_params_torch_to_tf(tf_enc.mixer.out_proj, torch_enc.mixer.out_proj)
    transfer_layernorm_params_torch_to_tf(tf_enc.ln, torch_enc.ln)

    out_tf = tf_enc(tf_inp, attention_mask=attn_mask, position_ids=tf_pids, past_key_value=past_key_values)

    with torch.no_grad():
        out_torch = torch_enc(tf_tensor_to_torch(tf_inp), position_ids=tf_tensor_to_torch(tf_pids))

    mse = torch.square(out_torch[0] - tf_tensor_to_torch(out_tf[0])).mean()
    print(f"MSE: {mse.item()}")

    # assert torch.allclose(tf_tensor_to_torch(tf.cast(out_tf[0], dtype=tf.float32)), out_torch[0], atol=atol)

    pass


def transfer_embedding_torch_to_tf(torch_embedding: nn.Embedding, tf_emb: keras.layers.Layer):
    w = torch_embedding.weight
    tf_emb.set_weights([w.detach().numpy().astype(np.float16)])
    pass



def torch_moondream_to_keras(model: nn.Module, variant):
    config = model.config

    phiModel = PhiModel(config)

    phiModel.build((None, None))

    tf.random.set_seed(1234)
    # inp = tf.ones((1, 10), dtype=tf.int64)
    inp_ids = tf.random.uniform((1, 10), maxval=50000, dtype=tf.int64)
    inp = phiModel.compute_embeddings(inp_ids)

    empty_cache = phiModel.empty_cache()

    phiModel(inp, empty_cache)

    if variant == "tf":
        ternary_module_fn = ScaledTernaryTf
    elif variant == "unpack":
        ternary_module_fn = ScaledTernaryUnpack
    elif variant == "matmul":
        ternary_module_fn = ScaledTernaryMatmul

    print("setting ternary layers")
    for keras_enc_layer, enc_layer in zip(phiModel.h, model.transformer.h):
        if isinstance(enc_layer.mlp.fc1, ScaledTernaryLinear):
            act = partial(keras.activations.gelu, approximate=True)
            enc_layer: PhiDecoderLayer
            tern_fc1 = ternary_module_fn(enc_layer.mlp.fc1.bias.shape[-1], activation=act)
            tern_fc2 = ternary_module_fn(enc_layer.mlp.fc2.bias.shape[-1])
            tern_out_proj = ternary_module_fn(enc_layer.mixer.out_proj.bias.shape[-1])
            tern_wqkv = ternary_module_fn(enc_layer.mixer.Wqkv.bias.shape[-1])
            # if enc_layer.mixer.layer_idx == 1:
            #     tern_wqkv.debug = True
            keras_enc_layer.mlp.fc1 = tern_fc1
            keras_enc_layer.mlp.fc2 = tern_fc2
            keras_enc_layer.mixer.out_proj = tern_out_proj
            keras_enc_layer.mixer.Wqkv = tern_wqkv

            print(f"set ternary block {enc_layer.mixer.layer_idx}")
            pass
        pass

    # phiModel.embd(inp)

    print("converting embedding")
    transfer_embedding_torch_to_tf(model.transformer.embd.wte, phiModel.embd.wte)

    embed_tf = phiModel.compute_embeddings(inp_ids) # re-compute correct embeddings
    inp = embed_tf
    embed_torch = model.transformer.embd(tf_tensor_to_torch(inp_ids))
    embed_mse = torch.square(tf_tensor_to_torch(embed_tf) - embed_torch).mean()
    print("embedding MSE:", embed_mse.item())

    print("converting transformer blocks")
    for enc_layer, tf_enc_layer in zip(model.transformer.h, phiModel.h):
        transfer_encoder_block_torch_to_tf(enc_layer, tf_enc_layer, empty_cache)
        print(f"converted block {enc_layer.mixer.layer_idx}")
        pass

    print("converting head")
    transfer_layernorm_params_torch_to_tf(phiModel.lm_head.ln, model.lm_head.ln)
    transfer_dense_params_torch_to_tf(phiModel.lm_head.linear, model.lm_head.linear)

    print("conversion done")

    # phiModel.h[1].mixer.Wqkv.debug = True # does nothing?

    print("inferring values with TF model")
    out_tf = phiModel(inp, empty_cache)[0]
    # out_tf = out_tf["logits"]

    print("inferring values with torch model")
    with torch.inference_mode():
        out_torch = model(input_ids=tf_tensor_to_torch(inp_ids))
        # out_torch = model(inputs_embeds=tf_tensor_to_torch(inp))

    diff = torch.square(tf_tensor_to_torch(out_tf) - out_torch["logits"]).sum()
    mse = torch.square(tf_tensor_to_torch(out_tf) - out_torch["logits"]).mean()

    print(f"error: {diff.item()}, MSE: {mse.item()}")

    return phiModel, out_torch, out_tf



