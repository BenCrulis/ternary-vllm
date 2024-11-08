from typing import Optional, Tuple, List, Union

import math

import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from larq.layers import QuantDense

from transformers.models.phi.modeling_phi import PhiConfig

import torch


def newGELUActivation(input):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * input * (1.0 + K.tanh(tf.sqrt(2.0 / math.pi) * (input + 0.044715 * K.pow(input, 3))))


ACT2FN = {
    # "gelu_new": newGELUActivation,
    "gelu_new": lambda x: keras.activations.gelu(x, approximate=True),
    "relu": "relu"
}


class PhiMLP(Layer):
    def __init__(self, config, dtype=tf.float32):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = Dense(config.intermediate_size, activation=self.activation_fn, dtype=dtype)
        self.fc2 = Dense(config.hidden_size, dtype=dtype)

    def call(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = tf.expand_dims(tf.gather_nd(cos, indices=position_ids[..., None]), axis=unsqueeze_dim)
    sin = tf.expand_dims(tf.gather_nd(sin, indices=position_ids[..., None]), axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PhiRotaryEmbedding(Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, dtype=tf.float32):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        with tf.device(device):
            inv_freq = 1.0 / (
                self.base ** (tf.cast(tf.range(0, self.dim, 2), dtype) / self.dim)
            )
        self.inv_freq = inv_freq # buffer

    def _compute_sin_cos(self, seq_len, device):
        with tf.device(device):
            t = tf.range(
                seq_len, dtype=self.inv_freq.dtype
            )

        freqs = t[:, None] * self.inv_freq[None, :]
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = tf.concat((freqs, freqs), axis=-1)
        return tf.cos(emb), tf.sin(emb)

    def call(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # if seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        sin, cos = self._compute_sin_cos(seq_len, x.device)
        return (
            tf.cast(sin, x.dtype),
            tf.cast(cos, x.dtype),
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def tf_attention(query, value, key, attn_mask=True, is_causal=False):
    L = tf.shape(query)[-2]
    S = tf.shape(key)[-2]
    scale_factor = 1.0 / tf.sqrt(tf.cast(tf.shape(query)[-1], dtype=tf.float32))
    # attn_bias = tf.zeros((L, S), dtype=query.dtype)
    attn_bias = attn_mask
    n_dim = len(key.shape)
    key_t = tf.transpose(key, [*[i for i in range(n_dim-2)], n_dim-1, n_dim-2])
    attn_weight = query @ key_t * scale_factor
    attn_weight += attn_bias
    attn_weight = keras.activations.softmax(attn_weight, axis=-1)
    return attn_weight @ value


def get_usable_length(past_key_value, new_seq_length, max_length=2000):
    previous_seq_length = tf.shape(past_key_value)[-2]
    if previous_seq_length + new_seq_length > max_length:
        return max_length - new_seq_length
    return previous_seq_length


class PhiAttention(Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PhiConfig, layer_idx: Optional[int] = None, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            print(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.Wqkv = Dense(
            3 * self.num_heads * self.head_dim, use_bias=True, dtype=dtype
        )
        self.out_proj = Dense(
            self.hidden_size, use_bias=True, dtype=dtype
        )

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = PhiRotaryEmbedding(
            int(self.partial_rotary_factor * self.head_dim),
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            dtype=self.dtype,
        )
       

    def call(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz = tf.shape(hidden_states)[0]
        q_len = tf.shape(hidden_states)[1]

        DEBUG_IDX = -1

        if self.layer_idx == DEBUG_IDX:
            tf.print(f"layer {self.layer_idx}")
            # tf.print("past key value", past_key_value)
            tf.print("hidden states", hidden_states)
            tf.print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

        query_states, key_states, value_states = tf.split(self.Wqkv(hidden_states), 3, axis=-1)

        if self.layer_idx == DEBUG_IDX:
            tf.print("initial key states", key_states)

        query_states = tf.transpose(tf.reshape(query_states, (
            bsz, q_len, self.num_heads, self.head_dim
        )), perm=(0, 2, 1, 3))
        key_states = tf.transpose(tf.reshape(key_states, (
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )), perm=(0, 2, 1, 3))
        value_states = tf.transpose(tf.reshape(value_states, (
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )), perm=(0, 2, 1, 3))

        kv_seq_len = tf.shape(key_states)[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += get_usable_length(past_key_value, kv_seq_len)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # print("cos", cos)
        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, num_heads, seq_length, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids
        )
        # print("query rot", query_rot)
        # [batch_size, num_heads, seq_length, head_dim]
        query_states = tf.concat((query_rot, query_pass), axis=-1)
        key_states = tf.concat((key_rot, key_pass), axis=-1)

        # cache_kwargs = {
        #     "sin": sin,
        #     "cos": cos,
        #     "partial_rotation_size": self.rotary_emb.dim,
        # }

        past_key_value = tf.concat([past_key_value[self.layer_idx], tf.stack([key_states, value_states], axis=0)], axis=-2)
        key_states, value_states = tf.unstack(past_key_value, 2, axis=0)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if self.layer_idx == DEBUG_IDX:
            tf.print("key states", key_states)
        # scaled_dot_product_attn_layer = keras.layers.Attention(use_scale=False)
        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states, key_states, value_states, attn_mask=attention_mask
        # )
        # attn_output = scaled_dot_product_attn_layer(
        #     [query_states, value_states, key_states], use_causal_mask=True
        # )
        attn_output = tf_attention(
            query_states, value_states, key_states, attn_mask=attention_mask
        )
        
        if self.layer_idx == DEBUG_IDX:
            tf.print("attn output raw", attn_output)
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))

        attn_output = self.out_proj(attn_output)
        if self.layer_idx == DEBUG_IDX:
            tf.print("out proj", attn_output)

        return attn_output, past_key_value, key_states, value_states


class PhiDecoderLayer(Layer):
    def __init__(self, config: PhiConfig, layer_idx: int, dtype=tf.float32):
        super().__init__()
        self.mixer = PhiAttention(
            config, layer_idx=layer_idx
        )
        self.mlp = PhiMLP(config, dtype=dtype)
        self.ln = LayerNormalization(epsilon=config.layer_norm_eps)

    def call(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.ln(hidden_states)
        # if self.mixer.layer_idx == 0:
        #     print("ln", hidden_states)
        # Self Attention
        attn_outputs, present_key_value, ks, vs = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        # print("attn output", attn_outputs)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # print("mlp", feed_forward_hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = hidden_states
        return outputs, present_key_value, ks, vs


class Embedding(Layer):
    def __init__(self, config: PhiConfig, dtype=tf.float32):
        super().__init__()
        self.wte = keras.layers.Embedding(
            config.vocab_size, config.hidden_size, # padding_idx=config.pad_token_id
            dtype=dtype,
        )

    def call(self, input_ids: tf.Tensor) -> tf.Tensor:
        return self.wte(input_ids)


class LMHead(Layer):
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.ln = keras.layers.LayerNormalization(center=True, scale=True)
        self.linear = keras.layers.Dense(config.vocab_size)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.ln(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    from argparse import Namespace
    from cache import DynamicCache

    from utils.conversion import transfer_dense_params_tf_to_torch, transfer_layernorm_params_tf_to_torch

    config = Namespace(
        **{
        "bos_token_id": 1,
        "embd_pdrop": 0.0,
        "eos_token_id": 2,
        "hidden_act": "gelu_new", # original
        # "hidden_act": "relu", # testing
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "phi",
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "num_key_value_heads": 32,
        "partial_rotary_factor": 0.5,
        "resid_pdrop": 0.0,
        "rope_scaling": None,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "transformers_version": "4.41.0",
        "use_cache": True,
        "vocab_size": 51200,
        # pytorch specific section
        "attention_dropout": 0.0, 
        "qk_layernorm": False,
        "_attn_implementation": "eager"
        }
    )
    phiMLP = PhiMLP(config)

    phiDec = PhiDecoderLayer(config, 0)

    a = tf.random.normal((1, 5, 2048))

    pos_ids = tf.range(0, 10)[None, :]

    dynamicCache = DynamicCache()
    # out = phiMLP(a)
    out = phiDec(a, past_key_value=dynamicCache, position_ids=pos_ids[:, :a.shape[1]])
    # out2 = phiDec(a*2, past_key_value=dynamicCache, position_ids=pos_ids[:, a.shape[1]:])

    print(out)

    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    pretrained_model_name_or_path = "vikhyatk/moondream2"
    class_reference = "vikhyatk/moondream2--modeling_phi.PhiDecoderLayer"

    TPhiDecLayer = get_class_from_dynamic_module(class_reference, pretrained_model_name_or_path)

    # from transformers.models.phi.modeling_phi import PhiDecoderLayer as TPhiDecLayer
    tphiDec = TPhiDecLayer(config, 0)

    transfer_dense_params_tf_to_torch(tphiDec.mlp.fc1, phiDec.mlp.fc1)
    transfer_dense_params_tf_to_torch(tphiDec.mlp.fc2, phiDec.mlp.fc2)
    transfer_dense_params_tf_to_torch(tphiDec.mixer.Wqkv, phiDec.mixer.Wqkv)
    transfer_dense_params_tf_to_torch(tphiDec.mixer.out_proj, phiDec.mixer.out_proj)
    transfer_layernorm_params_tf_to_torch(tphiDec.ln, phiDec.ln)

    out_t = tphiDec(torch.from_numpy(a.numpy()), position_ids=torch.from_numpy(pos_ids.numpy())[:, :a.shape[1]])

    pass



