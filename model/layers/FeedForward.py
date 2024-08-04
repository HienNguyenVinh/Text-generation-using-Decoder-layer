import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):
    '''
    input shape: [batch_size, seq_len, d_model]
    output shape: [batch_size, seq_len, d_model]
    '''

    def __init__(self, d_model, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.dense1 = tf.keras.layers.Dense(ff_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.dense1(inputs)  # [batch_size, seq_len, ff_dim]
        x = self.dense2(x)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        x = self.layer_norm(x + inputs)  # [batch_size, seq_len, d_model]

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })

        return config