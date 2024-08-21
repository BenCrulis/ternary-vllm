from typing import Any, List, Tuple, Optional, Union

from collections import namedtuple

import tensorflow as tf
from tensorflow import Tensor

from impl.tf.attention import AttentionMaskConverter


def prepare_4d_causal_attention_mask(
    input_shape: Union[Tuple, List],
    inputs_embeds: Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True)

    key_value_length = input_shape[-1] + past_key_values_length
    
    attention_mask = attn_mask_converter.to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
    )

    return attention_mask