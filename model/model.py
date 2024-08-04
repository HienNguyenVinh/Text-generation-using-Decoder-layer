import tensorflow as tf
import keras
from model.layers.PositionalEmbedding import PositionalEmbedding
from model.layers.Decoder import Decoder


class TextGenerationModel(tf.keras.Model):
    def __init__(self, *,
                 vocab_size: int,
                 input_size: int = None,
                 num_layers: int = 6,
                 d_model: int = 512,
                 num_heads: int = 8,
                 ff_dim: int = 2048,
                 seq_len: int = 32,
                 dropout_rate: float = 0.1
                 ):
        super().__init__()

        self.embed_input = PositionalEmbedding(vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoders = [Decoder(d_model, ff_dim, num_heads, dropout_rate) for _ in range(num_layers)]

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        x = self.embed_input(inputs)
        x = self.dropout(x)
        for layer in self.decoders:
            x = layer(x)

        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits