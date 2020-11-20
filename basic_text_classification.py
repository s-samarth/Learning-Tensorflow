import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Get dataset
# url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
#                                     untar=True, cache_dir='.',
#                                     cache_subdir='')

dataset_dir = os.path.join(os.path.dirname("Learning_Tensorflow"), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

# print(os.listdir(train_dir))
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#   print(f.read())

remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)


# Building The Dataset
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)

# Label 0 corresponds to neg
# Label 1 corresponds to pos

raw_val_ds = keras.preprocessing.text_dataset_from_directory(
  'aclImdb/train',
  batch_size=batch_size,
  validation_split=0.2,
  subset='validation',
  seed=seed
)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size,
    shuffle=False)


# Preprocessing the dataset (need not be a tf.function)
@tf.function()
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(max_tokens=max_features, 
                                  standardize=custom_standardization, 
                                  output_mode='int', 
                                  output_sequence_length=sequence_length)

# Make a text-only dataset without labels, then call adapt
train_text = raw_train_ds.map(lambda x, y : x)
vectorize_layer.adapt(train_text)

# (need not be a tf.function)
@tf.function()
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# Retrieve a batch from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized, Review", vectorize_text(first_review, first_label))

train_ds = raw_test_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Configuring the dataset for performance so that I/O isn't the bottleneck
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Making the Model, keras way
embedding_dim = 16

model = keras.Sequential([
  layers.Embedding(max_features+1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)
])

print(model.summary())

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
