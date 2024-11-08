from typing import Any, List, Tuple, Optional, Union

from collections import namedtuple

import tensorflow as tf
from tensorflow import Tensor

from tensorflow import keras
from tensorflow.keras.layers import Layer

from transformers.models.phi.modeling_phi import PhiConfig

from impl.tf.moondream.layers import PhiDecoderLayer, Embedding, LMHead
from impl.tf.moondream.cache import DynamicCache
from impl.tf.moondream.utils import prepare_4d_causal_attention_mask


class PhiModel(keras.Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """

    def __init__(self, config: PhiConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.input_spec = keras.layers.InputSpec(
        #     shape=(None, None),)

        self.embd = Embedding(config)
        self.h = [
            PhiDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.lm_head = LMHead(config)
    
    def build(self, input_shape):
        self.embd.build(input_shape),
        for h in self.h:
            h.build((*input_shape, 2048))
        self.lm_head.build((*input_shape, self.vocab_size))

    def get_input_embeddings(self):
        return self.embd.wte

    def set_input_embeddings(self, value):
        self.embd.wte = value

    # @tf.function
    def compute_embeddings(self, token_ids):
        return self.embd(token_ids)

    # @tf.function
    def empty_cache(self):
        return tf.zeros((0, len(self.h), 2, 1, 32, 64))

    # @tf.function
    def call(
        self,
        inputs_embeds: Optional[Tensor],
        past_key_values: Optional[List[Tensor]],
    ):
        batch_size = tf.shape(inputs_embeds)[0]
        seq_length = tf.shape(inputs_embeds)[1]
        past_key_values_length = tf.shape(past_key_values)[0]

        position_ids = tf.range(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=tf.int64,
        )
        position_ids = position_ids[None, ...]

        # 4d mask is passed through the layers
        attention_mask = prepare_4d_causal_attention_mask(
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_kv_states = []
        for decoder_layer in self.h:
            hidden_states, ks, vs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
            )

            all_kv_states.append(tf.stack([ks, vs], axis=0))
            pass
        all_kv_states = tf.stack(all_kv_states, axis=0)
        # layer, kv, batch, head, seq_length, head_features

        all_kv_states = tf.transpose(all_kv_states, perm=[4, 0, 1, 2, 3, 5])

        next_cache = all_kv_states

        logits = self.lm_head(hidden_states)
        
        return logits, next_cache