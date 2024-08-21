from typing import Any, List, Tuple, Optional, Union
from functools import partial

import numpy as np

from collections import namedtuple

import tensorflow as tf
from tensorflow import Tensor

from tensorflow import keras
from tensorflow.keras.layers import Layer

import larq
from larq.layers import QuantDense
from larq.quantizers import SteTern

import sys
sys.path += ["."]
# from custom_ops_mod import prevent_folding

from larq_compute_engine import ops


ternary_act = larq.utils.set_precision(2)(SteTern(0.5, ternary_weight_networks=False))


def py_prevent_folding(constant, witness):
    y = tf.numpy_function(lambda x, _: x, inp=[constant, witness], Tout=np.uint8, stateful=False, name="py_prevent_folding")
    return y


class ScaledTernary(Layer):
    def __init__(self, units, activation=None, clip_val=100.0):
        super().__init__()
        self.dense = QuantDense(units, kernel_quantizer=ternary_act, input_quantizer="ste_sign", use_bias=False)
        # self.scale = tf.ones((units,))
        self.scale = self.add_weight(shape=(units,), initializer="ones", trainable=True)
        self.bias = self.add_weight(shape=(units,), initializer="zeros", trainable=True)
        self.act = activation
        self.clip_val = tf.abs(clip_val)

    def call(self, x):
        # x = self.dense(x)
        if self.dense.weights == []:
            self.dense.build(x.shape)
        with larq.context.quantized_scope(True):
            x = x @ self.dense.weights[0]
        x = x * self.scale
        x = x + self.bias
        x = tf.clip_by_value(x, -self.clip_val, self.clip_val)
        if self.act is not None:
            x = self.act(x)
        return x


def pack_ternary(tensor: tf.Tensor):
    size = tensor.shape[-1]
    remainder = size % 4
    to_add = (4 - remainder) % 4
    
    if to_add != 0:
        tensor = tf.concat([tensor, tf.zeros((to_add,), dtype=tensor.dtype)], axis=-1)

    size = tensor.shape[-1]
    slice_size = size // 4
    
    packed = tf.zeros((slice_size,), dtype=tf.uint8)

    for i in range(4):
        s = tensor[i*slice_size:(i+1)*slice_size]
        val_pos = s == 1
        val_neg = s == -1
        val = tf.bitwise.bitwise_xor(tf.cast(val_pos, dtype=tf.uint8),
                                     tf.bitwise.left_shift(tf.cast(val_neg, dtype=tf.uint8), 1))
        shifted = tf.bitwise.left_shift(val, i*2)
        packed = tf.bitwise.bitwise_or(packed, shifted)
    return packed


# @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int8)])
def unpack_ternary_iterative_old(packed_tensor):
    packed_size = packed_tensor.shape[-1]
    tot_size = packed_tensor.shape * tf.constant(4, dtype=tf.int32) # each parameter is 2 bits so we can store 4 parameters per bytes
    unpacked = tf.zeros(tot_size)

    indices = tf.raw_ops.Range(start=tf.constant(0, dtype=tf.int32), limit=tf.reduce_sum(tot_size), delta=tf.constant(4, dtype=tf.int32))
    indices = tf.reshape(indices, (-1, 1))
    for i in range(4):
        shifted = tf.bitwise.right_shift(packed_tensor, tf.constant(i*2, dtype=tf.uint8))
        pos = tf.bitwise.bitwise_and(shifted, tf.constant(1, dtype=tf.uint8))
        neg = tf.bitwise.right_shift(tf.bitwise.bitwise_and(shifted, tf.constant(3, dtype=tf.uint8)), tf.constant(1, dtype=tf.uint8))
        shifted_indices = indices + i
        unpacked = tf.tensor_scatter_nd_add(unpacked, shifted_indices, tf.cast(pos, dtype=tf.float32))
        unpacked = tf.tensor_scatter_nd_sub(unpacked, shifted_indices, tf.cast(neg, dtype=tf.float32))
        pass
    
    return unpacked


def unpack_ternary_iterative(packed_tensor, dtype=tf.float32):
    res = []
    for i in range(4):
        shifted = tf.bitwise.right_shift(packed_tensor, tf.constant(i*2, dtype=tf.uint8))
        pos = tf.bitwise.bitwise_and(shifted, tf.constant(1, dtype=tf.uint8))
        neg = tf.bitwise.right_shift(tf.bitwise.bitwise_and(shifted, tf.constant(3, dtype=tf.uint8)), tf.constant(1, dtype=tf.uint8))
        res.append(tf.cast(pos, dtype=dtype) - tf.cast(neg, dtype=dtype))
        pass
    unpacked = tf.concat(res, axis=-1)
    return unpacked



def pack_ternary2(tensor: tf.Tensor):
    tensor = tf.reshape(tensor, shape=(-1,))
    size = tensor.shape[-1]
    remainder = size % 4
    to_add = (4 - remainder) % 4
    
    if to_add != 0:
        tensor = tf.concat([tensor, tf.zeros((to_add,), dtype=tensor.dtype)], axis=-1)

    size = tensor.shape[-1]
    slice_size = size // 4
    
    packed = tf.zeros((slice_size,), dtype=tf.uint8)

    for i in range(4):
        s = tensor[i*slice_size:(i+1)*slice_size]
        val_idx = s + 1
        val = tf.bitwise.left_shift(tf.cast(val_idx, dtype=tf.uint8), i*2)
        packed = tf.bitwise.bitwise_or(packed, val)
    return packed


def unpack_ternary_iterative2(packed_tensor, dtype=tf.float32):
    out1 = tf.bitwise.bitwise_and(tf.constant(0b00000011, dtype=tf.uint8), packed_tensor)
    out1 = tf.cast(out1, dtype=dtype) - tf.constant(1, dtype=dtype)


    out2 = tf.bitwise.bitwise_and(tf.constant(0b00001100, dtype=tf.uint8), packed_tensor)
    out2 = tf.bitwise.right_shift(out2, 2)
    out2 = tf.cast(out2, dtype=dtype) - tf.constant(1, dtype=dtype)


    out3 = tf.bitwise.bitwise_and(tf.constant(0b00110000, dtype=tf.uint8), packed_tensor)
    out3 = tf.bitwise.right_shift(out3, 4)
    out3 = tf.cast(out3, dtype=dtype) - tf.constant(1, dtype=dtype)


    out4 = tf.bitwise.bitwise_and(tf.constant(0b11000000, dtype=tf.uint8), packed_tensor)
    out4 = tf.bitwise.right_shift(out4, 6)
    out4 = tf.cast(out4, dtype=dtype) - tf.constant(1, dtype=dtype)

    unpacked = tf.concat([out1, out2, out3, out4], axis=-1)
    return unpacked


def unpack_ternary_iterative2_early_concat(packed_tensor, dtype=tf.float32):

    out1 = tf.bitwise.bitwise_and(tf.constant(0b00000011, dtype=tf.uint8), packed_tensor)

    out2 = tf.bitwise.bitwise_and(tf.constant(0b00001100, dtype=tf.uint8), packed_tensor)
    out2 = tf.bitwise.right_shift(out2, 2)

    out3 = tf.bitwise.bitwise_and(tf.constant(0b00110000, dtype=tf.uint8), packed_tensor)
    out3 = tf.bitwise.right_shift(out3, 4)

    out4 = tf.bitwise.bitwise_and(tf.constant(0b11000000, dtype=tf.uint8), packed_tensor)
    out4 = tf.bitwise.right_shift(out4, 6)

    unpacked = tf.concat([out1, out2, out3, out4], axis=-1)

    unpacked = tf.cast(unpacked, dtype=dtype) - tf.constant(1, dtype=dtype)
    return unpacked


class ScaledTernaryTf(Layer):
    def __init__(self, units, activation=None, clip_val=100.0):
        super().__init__()
        self.units = units
        self.scale = self.add_weight("scale", shape=(units,), initializer="ones", trainable=True)
        self.bias = self.add_weight("bias", shape=(units,), initializer="zeros", trainable=True)
        self.act = activation
        self.clip_val = tf.abs(clip_val)
        self.debug = False

        if True:
            self.pack_fn = pack_ternary2
            self.unpack_fn = unpack_ternary_iterative2
        else:
            self.pack_fn = pack_ternary
            self.unpack_fn = unpack_ternary_iterative


    def build(self, input_shape):
        orig_shape = (self.units, input_shape[-1])
        self.orig_shape = orig_shape
        tot_size = orig_shape[0] * orig_shape[1]
        self.tot_size = tot_size
        remainder = tot_size % 4
        final_size = tot_size // 4 + ((4 - remainder) if remainder != 0 else 0)

        def initializer(shape, dtype):
            import numpy as np
            return tf.convert_to_tensor(np.random.randint(0, 255, size=shape, dtype=np.uint8))

        self.pw = self.add_weight("pw", shape=(final_size,), initializer=initializer, dtype=tf.uint8, trainable=False)

    @tf.function
    def call(self, x):
        # b = tf.constant((8, 4, 2, 1), dtype=tf.int32)
        # useless_node = tf.cast(tf.reshape(x, [-1])[0], tf.uint8)
        if self.debug:
            div = "---------------------------------------------------------------------------------------------------"
            print(div)
            tf.print(div)
            tf.print("input x:", x)
        pw = tf.cond(tf.reshape(x, [-1])[0] < 5,
                lambda: self.pw,
                lambda: self.pw)
        # pw = prevent_folding(self.pw, x)
        # pw = py_prevent_folding(self.pw, x)
        # pw = tf.raw_ops.PlaceholderWithDefault(input=self.pw, shape=self.pw.shape)

        if self.debug:
            tf.print("pw:", pw)
        # pw = self.pw + useless_node * 0
        # pw = self.pw + tf.cast(tf.random.uniform((1,), minval=0.0, maxval=0.0), dtype=tf.uint8)
        # pw = tf.raw_ops.DebugIdentityV2(input=self.pw) # operation not recognized after export
        # pw = tf.raw_ops.Identity(input=self.pw) # Identity does not work
        
        # unpacked = tf.bitwise.bitwise_and(pw[:, None], b) > 0
        unpacked = self.unpack_fn(pw, dtype=x.dtype)

        if self.debug:
            tf.print("unpacked:", unpacked)
        
        w = tf.reshape(unpacked[:self.tot_size], self.orig_shape)
        x = tf.matmul(x, w, transpose_b=True)
        x = x * self.scale
        x = x + self.bias
        clip_val = tf.cast(self.clip_val, x.dtype)
        x = tf.clip_by_value(x, -clip_val, clip_val)
        if self.act is not None:
            x = self.act(x)

        if self.debug:
            tf.print("out x:", x)
        return x



class ScaledTernaryMatmul(Layer):
    def __init__(self, units, activation=None, clip_val=100.0):
        super().__init__()
        self.units = units
        self.scale = self.add_weight("scale", shape=(units,), initializer="ones", trainable=True)
        self.bias = self.add_weight("bias", shape=(units,), initializer="zeros", trainable=True)
        self.act = activation
        self.clip_val = tf.abs(clip_val)
        self.debug = False

    def build(self, input_shape):
        orig_shape = (self.units, input_shape[-1])
        self.orig_shape = orig_shape
        tot_size = orig_shape[1]
        self.tot_size = tot_size
        remainder = tot_size % 4
        final_size = (self.units, tot_size // 4 + ((4 - remainder) if remainder != 0 else 0))

        def initializer(shape, dtype):
            import numpy as np
            return tf.convert_to_tensor(np.random.randint(0, 255, size=shape, dtype=np.uint8))

        self.pw = self.add_weight("pw", shape=final_size, initializer=initializer, dtype=tf.uint8, trainable=False)

    @tf.function
    def call(self, x):
        orig_dim = x.shape[-1]
        x = ops.ternary_matmul(x, self.pw, self.scale, self.bias, tf.cast(self.clip_val, tf.float32))
        # x = ops.ternary_matmul(x, self.pw)

        # tf.print(f"orig_dim = {orig_dim}")
        # w = tf.ones(self.orig_shape, dtype=tf.uint8)
        # pw = tf.cond(tf.reshape(x, [-1])[0] < 5,
        #         lambda: self.pw,
        #         lambda: self.pw)
        # w = ops.unpack_ternary(self.pw, tf.constant(orig_dim, dtype=tf.int32))
        # w = tf.reshape(w, self.orig_shape)
        # w = tf.cast(w, dtype=tf.float32)
        # x = tf.matmul(x, w, transpose_b=True)

        # x = x * self.scale
        # x = x + self.bias
        # clip_val = tf.cast(self.clip_val, x.dtype)
        # x = tf.clip_by_value(x, -clip_val, clip_val)
        if self.act is not None:
            x = self.act(x)

        if self.debug:
            tf.print("out x:", x)
        return x


class ScaledTernaryUnpack(Layer):
    def __init__(self, units, activation=None, clip_val=100.0):
        super().__init__()
        self.units = units
        self.scale = self.add_weight("scale", shape=(units,), initializer="ones", trainable=True)
        self.bias = self.add_weight("bias", shape=(units,), initializer="zeros", trainable=True)
        self.act = activation
        self.clip_val = tf.abs(clip_val)
        self.debug = False

    def build(self, input_shape):
        orig_shape = (self.units, input_shape[-1])
        self.orig_shape = orig_shape
        tot_size = orig_shape[1]
        self.tot_size = tot_size
        remainder = tot_size % 4
        final_size = (self.units, tot_size // 4 + ((4 - remainder) if remainder != 0 else 0))

        def initializer(shape, dtype):
            import numpy as np
            return tf.convert_to_tensor(np.random.randint(0, 255, size=shape, dtype=np.uint8))

        self.pw = self.add_weight("pw", shape=final_size, initializer=initializer, dtype=tf.uint8, trainable=False)

    @tf.function
    def call(self, x):
        orig_dim = x.shape[-1]

        w = ops.unpack_ternary(self.pw, tf.constant(orig_dim, dtype=tf.int32))
        x = tf.matmul(x, w, transpose_b=True)

        x = x * self.scale
        x = x + self.bias
        clip_val = tf.cast(self.clip_val, x.dtype)
        x = tf.clip_by_value(x, -clip_val, clip_val)
        if self.act is not None:
            x = self.act(x)

        if self.debug:
            tf.print("out x:", x)
        return x


class ScaledTernaryPackedMM(Layer):
    def __init__(self, units, activation=None, clip_val=100.0):
        super().__init__()
        self.units = units
        self.scale = self.add_weight("scale", shape=(units,), initializer="ones", trainable=True)
        self.bias = self.add_weight("bias", shape=(units,), initializer="zeros", trainable=True)
        self.act = activation
        self.clip_val = tf.abs(clip_val)
        self.debug = False

    def build(self, input_shape):
        self.input_shp = input_shape
        orig_shape = (self.units, input_shape[-1])
        self.orig_shape = orig_shape
        remainder = orig_shape[-1] % 4
        final_size = orig_shape[-1] // 4 + ((4 - remainder) if remainder != 0 else 0)

        final_shape = (self.units, final_size)

        def initializer(shape, dtype):
            import numpy as np
            return tf.convert_to_tensor(np.random.randint(0, 255, size=shape, dtype=np.uint8))

        self.pw = self.add_weight("pw", shape=final_shape, initializer=initializer, dtype=tf.uint8, trainable=False)

    def pack_fn(self, w):
        size = w.shape[-1]
        remainder = size % 4
        to_add = (4 - remainder) % 4
        
        if to_add != 0:
            w = tf.concat([w, tf.zeros((to_add,), dtype=w.dtype)], axis=-1)

        size = w.shape[-1]
        slice_size = size // 4
        
        packed = tf.zeros((*w.shape[:-1], slice_size), dtype=tf.uint8)

        tensors = []

        for j in range(slice_size):
            s = w[..., j*4:(j+1)*4]
            val_idx = s + 1
            packed = tf.zeros(w.shape[:-1], dtype=tf.uint8)
            for i in range(4):
                val = tf.bitwise.left_shift(tf.cast(val_idx[:, i], dtype=tf.uint8), i*2)
                packed = tf.bitwise.bitwise_or(packed, val)
            tensors.append(packed)
        packed = tf.stack(tensors, -1)
        return packed

    # @tf.function
    def call_old(self, x):
        pw = tf.cond(tf.reshape(x, [-1])[0] < 5,
                lambda: self.pw,
                lambda: self.pw)
        
        dtype = x.dtype
        
        inp_shape = tf.shape(x)[:-1]

        if inp_shape.shape == 1:
            out = tf.zeros((inp_shape[0], self.orig_shape[0]), dtype=dtype)
        elif inp_shape.shape == 2:
            out = tf.zeros((inp_shape[0], inp_shape[1], self.orig_shape[0]), dtype=dtype)
        elif inp_shape.shape == 3:
            out = tf.zeros((inp_shape[0], inp_shape[1], inp_shape[2], self.orig_shape[0]), dtype=dtype)
        elif inp_shape.shape == 4:
            out = tf.zeros((inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3], self.orig_shape[0]), dtype=dtype)

        @tf.function
        def loop_body(i, out):
            packed_tensor = pw[:, i]
            out1 = tf.bitwise.bitwise_and(tf.constant(0b00000011, dtype=tf.uint8), packed_tensor)
            out1 = tf.cast(out1, dtype=dtype) - tf.constant(1, dtype=dtype)

            out2 = tf.bitwise.bitwise_and(tf.constant(0b00001100, dtype=tf.uint8), packed_tensor)
            out2 = tf.bitwise.right_shift(out2, 2)
            out2 = tf.cast(out2, dtype=dtype) - tf.constant(1, dtype=dtype)

            out3 = tf.bitwise.bitwise_and(tf.constant(0b00110000, dtype=tf.uint8), packed_tensor)
            out3 = tf.bitwise.right_shift(out3, 4)
            out3 = tf.cast(out3, dtype=dtype) - tf.constant(1, dtype=dtype)

            out4 = tf.bitwise.bitwise_and(tf.constant(0b11000000, dtype=tf.uint8), packed_tensor)
            out4 = tf.bitwise.right_shift(out4, 6)
            out4 = tf.cast(out4, dtype=dtype) - tf.constant(1, dtype=dtype)

            out = out + x[..., i*4, None] * out1
            out = out + x[..., i*4+1, None] * out2
            out = out + x[..., i*4+2, None] * out3
            out = out + x[..., i*4+3, None] * out4
            return i + 1, out

        # Define the loop condition
        def loop_cond(i, out):
            return i < pw.shape[-1] - 1

        i = tf.constant(0)
        _, out = tf.while_loop(loop_cond, loop_body, [i, out])

        last_idx = self.orig_shape[-1]
        remaining = last_idx % 4

        if remaining >= 1:
            packed_tensor = pw[:, last_idx // 4]
            out1 = tf.bitwise.bitwise_and(tf.constant(0b00000011, dtype=tf.uint8), packed_tensor)
            out1 = tf.cast(out1, dtype=dtype) - tf.constant(1, dtype=dtype)
            out = out + x[..., i*4, None] * out1

        if remaining >= 2:
            out2 = tf.bitwise.bitwise_and(tf.constant(0b00001100, dtype=tf.uint8), packed_tensor)
            out2 = tf.bitwise.right_shift(out2, 2)
            out2 = tf.cast(out2, dtype=dtype) - tf.constant(1, dtype=dtype)
            out = out + x[..., i*4+1, None] * out2

        if remaining >= 3:
            out3 = tf.bitwise.bitwise_and(tf.constant(0b00110000, dtype=tf.uint8), packed_tensor)
            out3 = tf.bitwise.right_shift(out3, 4)
            out3 = tf.cast(out3, dtype=dtype) - tf.constant(1, dtype=dtype)
            out = out + x[..., i*4+2, None] * out3

        if remaining >= 4:
            out4 = tf.bitwise.bitwise_and(tf.constant(0b11000000, dtype=tf.uint8), packed_tensor)
            out4 = tf.bitwise.right_shift(out4, 6)
            out4 = tf.cast(out4, dtype=dtype) - tf.constant(1, dtype=dtype)
            out = out + x[..., i*4+3, None] * out4
        
        out = out * self.scale + self.bias
        clip_val = tf.cast(self.clip_val, x.dtype)
        out = tf.clip_by_value(out, -clip_val, clip_val)
        if self.act is not None:
            out = self.act(out)

        return out
    
    def call(self, x):
        pw = tf.cond(tf.reshape(x, [-1])[0] < 5,
                lambda: self.pw,
                lambda: self.pw)
        
        dtype = x.dtype
        
        inp_shape = tf.shape(x)[:-1]

        dim_to_ignore = 1
        if inp_shape.shape == 1:
            out = tf.zeros((inp_shape[0], self.orig_shape[0]), dtype=dtype)
        elif inp_shape.shape == 2:
            dim_to_ignore = 2
            out = tf.zeros((inp_shape[0], inp_shape[1], self.orig_shape[0]), dtype=dtype)
        elif inp_shape.shape == 3:
            dim_to_ignore = 3
            out = tf.zeros((inp_shape[0], inp_shape[1], inp_shape[2], self.orig_shape[0]), dtype=dtype)
        elif inp_shape.shape == 4:
            dim_to_ignore = 4
            out = tf.zeros((inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3], self.orig_shape[0]), dtype=dtype)

        out1 = tf.bitwise.bitwise_and(tf.constant(0b00000011, dtype=tf.uint8), pw)
        out1 = tf.cast(out1, dtype=dtype) - tf.constant(1, dtype=dtype)

        out2 = tf.bitwise.bitwise_and(tf.constant(0b00001100, dtype=tf.uint8), pw)
        out2 = tf.bitwise.right_shift(out2, 2)
        out2 = tf.cast(out2, dtype=dtype) - tf.constant(1, dtype=dtype)

        out3 = tf.bitwise.bitwise_and(tf.constant(0b00110000, dtype=tf.uint8), pw)
        out3 = tf.bitwise.right_shift(out3, 4)
        out3 = tf.cast(out3, dtype=dtype) - tf.constant(1, dtype=dtype)

        out4 = tf.bitwise.bitwise_and(tf.constant(0b11000000, dtype=tf.uint8), pw)
        out4 = tf.bitwise.right_shift(out4, 6)
        out4 = tf.cast(out4, dtype=dtype) - tf.constant(1, dtype=dtype)

        to_pad = 4 - (tf.shape(x)[-1] % 4)

        if to_pad != 0 and to_pad != 4:
            x = tf.pad(x, (*[(0, 0) for _ in range(dim_to_ignore)], (0, to_pad)))

        out = out + x[..., ::4] @ tf.transpose(out1)
        out = out + x[..., 1::4] @ tf.transpose(out2)
        out = out + x[..., 2::4] @ tf.transpose(out3)
        out = out + x[..., 3::4] @ tf.transpose(out4)

        out = out * self.scale + self.bias
        clip_val = tf.cast(self.clip_val, x.dtype)
        out = tf.clip_by_value(out, -clip_val, clip_val)
        if self.act is not None:
            out = self.act(out)

        return out

    

class TestBinary(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform")
    
    def call(self, x):
        x = x @ larq.quantizers.ste_sign(self.kernel + tf.reduce_sum(x) * 0)
        return x


class TestBinary2(Layer):
    def __init__(self, units):
        super().__init__()
        self.dense = QuantDense(units, kernel_quantizer="ste_sign", input_quantizer="ste_sign", use_bias=False)

    def call(self, x):
        if self.dense.weights == []:
            self.dense.build(x.shape)
        x = x @ larq.quantizers.ste_sign(self.dense.weights[0] + tf.reduce_sum(x) * 0)
        return x
    

class TestBinary3(Layer):
    def __init__(self, units):
        super().__init__()
        self.dense = QuantDense(units, kernel_quantizer="ste_sign", input_quantizer="ste_sign", use_bias=False)

    def call(self, x):
        if self.dense.weights == []:
            self.dense.build(x.shape)
        x = x @ larq.quantizers.ste_sign(self.dense.weights[0] + tf.reduce_sum(x) * 0)
        return x


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int8)], autograph=False)
def no_op(tensor):
    return tensor + 0


class TestBinary4(Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.input_size = input_shape[-1]
        assert input_shape[-1] * self.units % 8 == 0
        compressed_size = int(input_shape[-1] * self.units / 8)
        self.kernel = self.add_weight(shape=(compressed_size,), initializer="ones", dtype=tf.int8)

    def call(self, x):
        # compressed_weights = self.kernel + tf.cast(x[0,0], dtype=tf.int8) * 0
        compressed_weights = self.kernel
        # kernel = tf.zeros((self.units, self.input_size))
        tensors = []
        for i in range(8):
            # kernel[i::8] = compressed_weights
            tensors.append(compressed_weights)
        
        kernel = tf.stack(tensors)
        kernel = tf.cast(kernel, tf.float32)
        kernel = tf.reshape(kernel, (self.input_size, self.units))
        x = x @ kernel
        # x = x + tf.reduce_sum(kernel)
        return x


if __name__ == "__main__":
    import larq_compute_engine as lce

    # tf.config.optimizer.set_experimental_options({
    #     "constant_folding": False,
    #     "disable_model_pruning": True,
    #     "remapping": False,
    #     })

    # quantDense = larq.layers.QuantConv2D(1000, 1, kernel_quantizer="ste_sign", input_quantizer="ste_sign", use_bias=False)
    # quantDense = TestBinary4(1000)
    # tf.compat.v1.enable_control_flow_v2()
    
    import numpy as np
    np.random.seed(1234)
    tf.random.set_seed(1234)

    quantDense = ScaledTernaryPacked(1000)
    # quantDense = ScaledTernaryPackedMM(1000)
    # quantDense = keras.layers.Dense(1000)
    test_input_shape = (503,)
    out = quantDense(tf.ones((3, 5, *test_input_shape)))
    print(out)

    inputs = keras.Input(test_input_shape)
    outputs = quantDense(inputs)
    quantDense = keras.Model(inputs=inputs, outputs=outputs)
    tf.saved_model.save(quantDense, "/tmp/quantDense")

    converter = tf.lite.TFLiteConverter.from_saved_model("/tmp/quantDense")
    # converter = tf.lite.TFLiteConverter.from_keras_model(quantDense)
    # converter.experimental_new_converter = False
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]

    converted = converter.convert()

    with open("test_tf.tflite", "wb") as f:
        print("writing tflite model to disk")
        f.write(converted)
    

    with larq.context.quantized_scope(True):
        inp_quant = keras.Input(test_input_shape)
        out_quant = quantDense(inp_quant)
        quantModelTest = keras.Model(inputs=inp_quant, outputs=out_quant)
        larq.models.summary(quantModelTest)

        print("converting larq keras test model to tflite")
        converted = lce.convert_keras_model(quantModelTest)
        with open("test.tflite", "wb") as f:
            print("writing tflite model to disk")
            f.write(converted)

    pass

