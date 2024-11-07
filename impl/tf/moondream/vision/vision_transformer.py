import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from keras.api.activations import gelu


# This is for typehinting and intllisense
import tensorflow.python.keras as _tfk
import tensorflow.python.keras.layers as _layers

# This gets highlighted as error by my linter, but it runs
keras: _tfk
layers: _layers


gelu_ = lambda x: gelu(x, approximate=True)


class Attention(layers.Layer):

    def __init__(self, dim, num_heads=16):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)


    def call(self, x: tf.Tensor) -> tf.Tensor:
        B, N, C = x.shape
        qkv = tf.transpose(tf.reshape(self.qkv(x), (B, N, 3, self.num_heads, self.head_dim)), (2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv, axis=0)

        # x = F.scaled_dot_product_attention(q, k, v)
        x = tf.einsum("bhnd,bhkd->bhnk", q, k)
        x = x / (self.head_dim ** 0.5)
        x = tf.nn.softmax(x, axis=-1)
        x = tf.einsum("bhnk,bhkd->bhnd", x, v)
        # check the above code block

        # x = x.transpose(1, 2).reshape(B, N, C)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))

        x = self.proj(x)
        return x


class VitBlock(layers.Layer):

    def __init__(self, embed_dim):
        super().__init__()
        self.attn = Attention(embed_dim)
        self.mlp = MLP(embed_dim, 4304)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(keras.Model):

    def __init__(self):
        super().__init__()

        embed_len = 729
        embed_dim = 1152

        self.patch_embed = LinearPatchEmbedding()
        # self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_embed = self.add_weight(shape=(1, embed_len, embed_dim), initializer="random_normal", trainable=True)

        self.blocks = keras.Sequential(
            layers=[VitBlock(embed_dim) for _ in range(27)]
        )
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)


class EncoderWrapper(keras.Model):

    def __init__(self):
        super().__init__()
        # self.model = nn.ModuleDict({"visual": VisionTransformer(use_flash_attn)})
        self.model = {"visual": VisionTransformer()}

    def call(self, x):
        return self.model["visual"](x)


class LinearPatchEmbedding(layers.Layer):

    def __init__(self):
        super().__init__()
        self.linear = layers.Dense(1152) #nn.Linear(588, 1152)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        b, c, hp1, wp2 = x.shape
        p1, p2 = 14, 14
        h, w = hp1 // p1, wp2 // p2
        x = tf.reshape(x, (b, c, h, p1, w, p2))
        x = tf.transpose(x, (0, 2, 4, 1, 3, 5))
        x = tf.reshape(x, (b, h * w, c * p1 * p2))

        return self.linear(x)


class MLP(layers.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = layers.Dense(hidden_features, activation=gelu_)
        self.fc2 = layers.Dense(out_features)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class VisionProjection(layers.Layer):
    def __init__(self):
        super().__init__()

        image_embedding_dim = 1152
        model_dim = 2048
        hidden_dim = model_dim * 4

        self.mlp = MLP(image_embedding_dim, hidden_dim, model_dim)

    def call(self, x):
        return self.mlp(x)


class VisionEncoder(layers.Layer):

    def __init__(self):
        super().__init__()

        self.encoder = EncoderWrapper()
        self.projection = VisionProjection()

        # self.preprocess = Compose(
        #     [
        #         Resize(size=(378, 378), interpolation=InterpolationMode.BICUBIC),
        #         ToImage(),
        #         ToDtype(torch.float32, scale=True),
        #         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     ]
        # )
        # equivalent in tensorflow
        # self.preprocess = keras.Sequential(
        #     [
        #         layers.experimental.preprocessing.Resizing(378, 378),
        #         layers.experimental.preprocessing.Rescaling(1.0 / 255),
        #         layers.experimental.preprocessing.Normalization(mean=[0.5, 0.5, 0.5], variance=[0.5, 0.5, 0.5]),
        #     ]
        # )


    def call(self, images) -> tf.Tensor:
        # squeeze first dimension
        x = tf.transpose(images, perm=[0, 3, 1, 2])
        x = self.encoder(x)
        x = self.projection(x)

        return x
