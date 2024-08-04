import warnings
warnings.filterwarnings('ignore')
from data import PrepareData, load_data, clean_text
import tensorflow as tf
from model import TextGenerationModel
from model.CustomLR import CustomSchedule
import os
from argparse import ArgumentParser

print('GPU is ', 'available' if tf.config.experimental.list_physical_devices('GPU') else 'not available')
print('Tensorflow version: ', tf.__version__)


parser = ArgumentParser()
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--vocab-size', default=21000, type=int)
parser.add_argument('--num-layers', default=6, type=int)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--ff-dim', default=1024, type=int)
parser.add_argument('--seq-len', default=64, type=int)
parser.add_argument('--dropout-rate', default=0.1, type=float)
parser.add_argument('--epochs', default=3, type=int)
args = parser.parse_args()

batch_size = args.batch_size
vocab_size = args.vocab_size
num_layers = args.num_layers
d_model = args.d_model
num_heads = args.num_heads
ff_dim = args.ff_dim
seq_len = args.seq_len
dropout_rate = args.dropout_rate
EPOCHS = args.epochs


print('Loading data...')
df = load_data()
df['content'] = df['content'].apply(clean_text)

dataset = PrepareData()
vocabulary, train_ds, val_ds = dataset.prepare_data(df.content.tolist(), batch_size, seq_len, vocab_size)
print('Data is Ready')

# Training model


model = TextGenerationModel(vocab_size=vocab_size,
                           num_layers=num_layers,
                           d_model=d_model,
                           num_heads=num_heads,
                           ff_dim=ff_dim,
                           seq_len=seq_len,
                           dropout_rate=dropout_rate)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9,
)

# Define loss and acc
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


checkpoint_filepath = '/checkpoint/checkpoint.weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_filepath)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


class EpochTrackerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global last_epoch
        last_epoch = epoch


epoch_tracker_callback = EpochTrackerCallback()

callbacks_list = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=False,
        save_freq='epoch'
    ),
    epoch_tracker_callback
]



model.compile(optimizer=optimizer,
              loss=masked_loss,
              metrics=[masked_accuracy])

history = model.fit(train_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_ds)