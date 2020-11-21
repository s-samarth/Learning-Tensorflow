import numpy as np 
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

X_train.shape = (60000, 28, 28)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
# output shape is (None, 10) ie batch size, 10

# logits is the inverse of sigmoid
logits = model(X_train[:1]).numpy()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1)
model.evaluate(X_test, y_test, verbose=2)

probab_model = keras.Sequential([
    model,
    keras.layers.Softmax()
])
