from generator import Generator
from model import TextGenerationModel
from train import vocab_size, num_layers, d_model, num_heads, ff_dim, seq_len, dropout_rate
import tensorflow as tf
import json

model = TextGenerationModel(vocab_size=vocab_size,
                           num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           ff_dim=ff_dim,
                           seq_len=seq_len,
                           dropout_rate=dropout_rate)

checkpoint_filepath = '/checkpoint/checkpoint.weights.h5'

model.load_weights(checkpoint_filepath)

vectorizer_path = 'vectorizer/vectorizer.json'
vocab_path = 'vectorizer/vocab.json'

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# Load the vectorizer configuration and weights
with open(vectorizer_path, 'r', encoding='utf-8') as f:
    vectorizer_info = json.load(f)
    config = vectorizer_info['config']
    weights = vectorizer_info['weights']

vectorizer = tf.keras.layers.TextVectorization.from_config(config)
vectorizer.set_weights([tf.constant(w) for w in weights])

max_new_tokens = 50
temperature = 0.9
generator = Generator(
    vectorizer, vocab, model, max_new_tokens, seq_len, temperature,
)


sentence = 'tôi thích'
generated_text, generated_tokens = generator(sentence)

print(sentence + ' ' + generated_text)
