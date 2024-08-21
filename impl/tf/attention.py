from typing import Optional, Union, List

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class AttentionMaskConverter:

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool):
        self.is_causal = is_causal

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: tf.DType,
    ) -> Optional[tf.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        # causal_4d_mask = None
        # if input_shape[-1] > 1 or self.sliding_window is not None:
        causal_4d_mask = self._make_causal_mask(
            input_shape,
            dtype,
            past_key_values_length=past_key_values_length,
        )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: tf.Tensor,
        query_length: int,
        dtype: tf.DType,
        key_value_length: Optional[int] = None,
    ) -> tf.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (tf.shape(attention_mask_2d)[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                past_key_values_length=past_key_values_length,
            )
       
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: List[int],
        dtype: tf.DType,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = tf.fill((tgt_len, tgt_len), dtype.min)
        mask_cond = tf.range(tf.shape(mask)[-1])
        # mask.masked_fill_(mask_cond < (mask_cond + 1).reshape(mask.shape[-1], 1), 0)
        mask = tf.where(mask_cond < tf.reshape((mask_cond + 1), (tf.shape(mask)[-1], 1)), tf.zeros((1,), dtype=dtype), mask)

        mask = tf.cast(mask, dtype)

        mask = tf.concat([tf.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

        return tf.broadcast_to(mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length))

    @staticmethod
    def _expand_mask(mask: tf.Tensor, dtype: tf.DType, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(tf.bool), tf.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: tf.Tensor,
        min_dtype: float,
    ):
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        # fmt: on
        if expanded_mask.dtype == tf.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )

        return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim=-1, keepdim=True))

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: Optional[tf.Tensor],
        inputs_embeds: tf.Tensor,
        past_key_values_length: int,
        sliding_window: Optional[int] = None,
        is_training: bool = False,
    ) -> bool:
        """
        Detects whether the optional user-specified attention_mask & the automatically created causal mask can be ignored in case PyTorch's SDPA is used, rather relying on SDPA's `is_causal` argument.

        In case no token is masked in the `attention_mask` argument, if `query_length == 1` or
        `key_value_length == query_length`, we rather rely on SDPA `is_causal` argument to use causal/non-causal masks,
        allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
        """

        _, query_length = tf.shape(inputs_embeds)[0], tf.shape(inputs_embeds)[1]
        key_value_length = query_length + past_key_values_length

        is_tracing = (
            torch.jit.is_tracing()
            or isinstance(inputs_embeds, torch.fx.Proxy)
            or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
        )

        ignore_causal_mask = False

        if attention_mask is None:
            # TODO: When tracing with TorchDynamo with fullgraph=True, the model is recompiled depending on the input shape, thus SDPA's `is_causal` argument is rightfully updated (see https://gist.github.com/fxmarty/1313f39037fc1c112508989628c57363). However, when using `torch.export` or
            # or `torch.onnx.dynamo_export`, we must pass an example input, and `is_causal` behavior is hard-coded. If a user exports a model with q_len > 1, the exported model will hard-code `is_causal=True` which is in general wrong (see https://github.com/pytorch/pytorch/issues/108108).
            # Thus, we only set `ignore_causal_mask = True` if the model is set to training.
            #
            # Besides, jit.trace can not handle the `q_len > 1` condition for `is_causal` (`TypeError: scaled_dot_product_attention(): argument 'is_causal' must be bool, not Tensor`).
            if (
                (is_training or not is_tracing)
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif (is_training or not is_tracing) and torch.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    # For query_length == 1, causal attention and bi-directional attention are the same.
                    ignore_causal_mask = True

                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                # TODO: maybe revisit this with https://github.com/pytorch/pytorch/pull/114823 in PyTorch 2.3.

        return ignore_causal_mask