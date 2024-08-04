import tensorflow as tf


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=10000, d_model=512, **kwargs):
        '''
        Args:
            seq_len: input length
            vocab size: input dim
            d_model: embedding vector size
        '''
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model, mask_zero=True)
        self.position_embeddings = self.positional_encoding(length=2048)

    def positional_encoding(self, n=10000, length=2048):
        assert self.d_model % 2 == 0
        d2 = self.d_model / 2

        position = np.arange(length)[:, np.newaxis]  # token position in sentence ~ [seq, 1]
        index = np.arange(d2)[np.newaxis, :]  # index position in embeded vector ~ [1, index]

        angle_rads = position * (np.power(n, -index / d2))  # p/(10000^(2i/d_model))

        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        length = tf.shape(inputs)[1]

        x = self.token_embeddings(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos_encoding = self.position_embeddings[tf.newaxis, :length, :]

        return x + pos_encoding

    def compute_mask(self, *args, **kwargs):
        return self.token_embeddings.compute_mask(*args, **kwargs)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "seq_len": self.seq_len,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        })