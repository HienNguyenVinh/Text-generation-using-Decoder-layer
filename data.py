import pandas as pd
import numpy as np
import re
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from tqdm import tqdm
import json


# Load data
def load_data():
    categories = ['chuyen-tham-kin', 'giao-duc', 'khoa-hoc-cong-nghe', 'lich-su', 'phat-trien-ban-than', 'quan-diem-tranh-luan', 'sach', 'tai-chinh', 'tam-ly-hoc', 'yeu']

    df_list = []
    for c in categories:
        tmp_df = pd.read_csv(f'/data/2000-blogs-on-spiderum/{c}.csv')
        df_list.append(tmp_df)

    df = pd.concat(df_list, ignore_index=True)

    return df


# Clean text
def clean_text(texts):
    unwanted_patterns = [
        'spiderum.com', '=', '\xa0', '\n', '<', '>', '-', '_', '<3', '(', ')', ':', '*', '/', '@', '+'
    ]
    combined_pattern = '|'.join(map(re.escape, unwanted_patterns))

    cleaned_texts = re.sub(combined_pattern, ' ', texts)
    cleaned_texts = re.sub(r'\.{2,}', '.', cleaned_texts)
    cleaned_texts = re.sub(r'(?<=[a-zA-Z])([.!?])', r'\1 ', cleaned_texts)
    cleaned_texts = re.sub(r'\s+', ' ', cleaned_texts).strip()

    return cleaned_texts


# Prepare data
class PrepareData:
    def __init__(self):
        self.vectorizer = None

    def split_data(self, texts, ratio=0.9):
        n = int(ratio * len(texts))
        train_texts = texts[:n]
        val_texts = texts[n:]

        return train_texts, val_texts

    def build_tokenizer(self, texts, vocab_size):
        self.vectorizer = TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            standardize="lower_and_strip_punctuation",
            output_sequence_length=None
        )

        self.vectorizer.adapt(texts)

    def save_vectorizer(self):
        vectorizer_path = 'vectorizer/vectorizer.json'
        vocab_path = 'vectorizer/vocab.json'

        # Save the vocabulary
        vocab = self.vectorizer.get_vocabulary()
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f)

        # Save the vectorizer
        config = self.vectorizer.get_config()
        weights = self.vectorizer.get_weights()
        vectorizer_info = {
            'config': config,
            'weights': [weight.tolist() for weight in weights]
        }
        with open(vectorizer_path, 'w', encoding='utf-8') as f:
            json.dump(vectorizer_info, f)

    def tokenize(self, texts):
        tokenized_texts = []
        for text in texts:
            tmp = self.vectorizer([text])[0].numpy()
            tokenized_texts.append(tmp)

        return tokenized_texts

    def create_seq_pair(self, texts, seq_len):
        input_sequences = []
        target_sequences = []

        for text in tqdm(texts):
            for i in range(len(text) - seq_len):
                input_seq = text[i:i + seq_len]
                target_seq = text[i + 1:i + seq_len + 1]
                input_sequences.append(input_seq)
                target_sequences.append(target_seq)

        input_seqs = np.array(input_sequences)
        target_seqs = np.array(target_sequences)

        return input_seqs, target_seqs

    def create_dataset(self, text_pair, batch_size=64):
        text_dataset = tf.data.Dataset.from_tensor_slices(text_pair)

        # Split data into batches
        text_dataset = text_dataset.batch(batch_size)
        # Shuffle and prefetch to increase performance
        text_dataset = text_dataset.shuffle(10000).prefetch(tf.data.experimental.AUTOTUNE)

        return text_dataset

    def prepare_data(self, texts, batch_size, seq_len, vocab_size):
        self.build_tokenizer(texts, vocab_size)
        self.save_vectorizer()

        vectorized_texts = self.tokenize(texts)
        vectorized_train_texts, vectorized_val_texts = self.split_data(vectorized_texts)

        train_inputs, train_targets = self.create_seq_pair(vectorized_train_texts, seq_len)
        val_inputs, val_targets = self.create_seq_pair(vectorized_val_texts, seq_len)

        train_ds = self.create_dataset((train_inputs, train_targets), batch_size)
        val_ds = self.create_dataset((val_inputs, val_targets), batch_size)

        return self.vectorizer.get_vocabulary(), train_ds, val_ds

