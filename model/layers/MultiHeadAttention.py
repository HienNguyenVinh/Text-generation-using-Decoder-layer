import tensorflow as tf

def shape_list(x):
    # Deal with dynamic shape in tensorflow cleanly.
    '''
    e.g: x = tf.placeholder(tf.float32, shape=[None, 128, None])
        => static_shape(x) = [None, 128, None]
           dynamic_shape(x) = [batch_size, 128, seq_len]
    -> Ensures that tensor operations work correctly regardless of whether the input shape changes or not.
    '''
    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    return [dynamic_shape[i] if s is None else s for i, s in enumerate(static_shape)]


# Causal masking
def attention_mask(target_len, source_len, dtype):
    i = tf.range(target_len)[:, None]  # (target_len, 1)
    j = tf.range(source_len)  # (1, source_len)
    m = i >= j - source_len + target_len  # m ~ (target_len, source_len)

    return tf.cast(m, dtype)


def mask_attn_weights(w):
    # w ~ [batch_size, num_heads, target_sequence_length, source_sequence_length]
    # Information flows from source to target.
    _, _, target_len, source_len = shape_list(w)
    mask = attention_mask(target_len, source_len, dtype=w.dtype)
    mask = tf.reshape(mask, [1, 1, target_len, source_len])
    w = w * mask - tf.cast(1e10, w.dtype) * (1 - mask)

    return w


def scaled_dot_product_attention(q, k, v, use_causal_mask=False):
    '''
        q ~ [batch_size, num_heads, target_seq_len, head_dim]
        k ~ [batch_size, num_heads, source_seq_len, head_dim]
        v ~ [batch_size, num_heads, source_seq_len, head_dim]
    '''
    d_k = tf.cast(tf.shape(k)[-1], tf.float32)  # size of vector k ~ head_dim
    scores = tf.matmul(q, k,
                       transpose_b=True)  # Matmul of q and kT ~ [batch_size, num_heads, target_seq_len, source_seq_len]
    scaled_scores = scores / tf.math.sqrt(d_k)  # scale
    if use_causal_mask:
        scaled_scores = mask_attn_weights(scaled_scores)

    weights = tf.nn.softmax(scaled_scores, axis=-1)  # Softmax
    output = tf.matmul(weights, v)  # Matmul of Softmax and v

    return output


class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    input shape: [batch_size, seq_len, d_model]
    output shape: [batch_size, seq_len, d_model]
    '''

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads

        if d_model & num_heads != 0:
            raise ValueError(
                f"Dimension of the embedding space = {embed_dim} should be divisible by number of heads = {num_heads}")

        self.q_linear = tf.keras.layers.Dense(d_model)
        self.k_linear = tf.keras.layers.Dense(d_model)
        self.v_linear = tf.keras.layers.Dense(d_model)
        self.concat_linear = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # x ~ (batch_size, seq_len, d_model)
        head_dim = self.d_model // self.num_heads
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads,
                                 head_dim))  # -1 ~ Used to let TensorFlow automatically infer the remaining dimension so that the total number of elements does not change

        return tf.transpose(x, perm=[0, 2, 1, 3])  # ~ (batch_size, num_heads, seq_len, head_dim)

    def concat_heads(self, x, batch_size):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, q, k, v, use_causal_mask=False):
        batch_size = tf.shape(k)[0]
        q = self.q_linear(q)
        k = self.q_linear(k)
        v = self.q_linear(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention = scaled_dot_product_attention(q, k, v, use_causal_mask)
        concat = self.concat_heads(attention, batch_size)
        concat = self.concat_linear(concat)

        return concat

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "d_model": self.d_model,
            "h": self.num_heads,
        })
        return config