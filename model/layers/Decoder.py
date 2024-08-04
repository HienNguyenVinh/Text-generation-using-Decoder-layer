import tensorflow as tf
from model.layers.FeedForward import FeedForward
from model.layers.MultiHeadAttention import MultiHeadAttention


class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, ff_dim, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.causal_self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model, ff_dim, dropout_rate)

    def call(self, inputs):
        x = self.layer_norm_1(inputs + self.causal_self_attention(q=inputs, k=inputs, v=inputs, use_causal_mask=True))
        x = self.layer_norm_2(x + self.feed_forward(x))

        return x