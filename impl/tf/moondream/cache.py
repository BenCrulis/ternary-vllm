from typing import Tuple, Dict, Optional, Any, List

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class DynamicCache:
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[tf.Tensor] = []
        self.value_cache: List[tf.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[tf.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: tf.Tensor,
        value_states: tf.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = tf.concat([self.key_cache[layer_idx], key_states], axis=-2)
            self.value_cache[layer_idx] = tf.concat([self.value_cache[layer_idx], value_states], axis=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[tf.Tensor], Tuple[tf.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[tf.Tensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


if __name__ == "__main__":
    import keras
    class LayerWithCache(keras.Model):
        def __init__(self, units=100, **kwargs):
            super().__init__(**kwargs)
            self.a = tf.random.normal((units,))

        @tf.function
        def call(self, x, cache):
            x = self.a + x
            if cache is not None:
                updated_cache = x + cache["a"] + cache["b"]
                return x, updated_cache
            else:
                return x

        @tf.function
        def initial_cache(self, x):
            return tf.zeros(self.a.shape) + x * 0.0
        
        @tf.function
        def initial_cache_noarg(self):
            return {"a": tf.zeros(self.a.shape), "b": tf.zeros(self.a.shape)}

    mod = LayerWithCache()

    inp_cache_concr = {"a": tf.zeros((100,)), "b": tf.zeros((100,))}
    mod(tf.ones((100,)), mod.initial_cache_noarg())

    inp = keras.Input((100,))
    cache_inp = {"a": keras.Input((100,)), "b": keras.Input((100,))}

    out, out_cache = mod(inp, cache_inp)

    model = keras.Model(inputs=[inp, cache_inp], outputs=[out, out_cache])

    # converter = tf.lite.TFLiteConverter.from_keras_model(mod)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [
            mod.call.get_concrete_function(tf.zeros((100,)), inp_cache_concr),
            # mod.call.get_concrete_function(tf.zeros((100,)), None),
            mod.initial_cache.get_concrete_function(0.0),
            mod.initial_cache_noarg.get_concrete_function()
        ], mod)
    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS, # enable TensorFlow ops.
        # tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("converting model to TFLite")
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    print(interpreter.get_signature_list())

    init_cache = interpreter.get_signature_runner("initial_cache_noarg")

    pass
