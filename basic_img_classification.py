import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualizing the Images
# plt.figure()
# plt.imshow(train_imgs[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_imgs = train_imgs[..., tf.newaxis].astype('float32')
test_imgs = test_imgs[..., tf.newaxis].astype('float32')

# Dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels)).batch(32)

# Model
class LeNet(Model):
    def __init__(self):
        super().__init__()
        self.preprocess = keras.layers.experimental.preprocessing.Rescaling(1/255.)
        self.conv1 = keras.layers.Conv2D(6, (5, 5), activation='tanh')
        self.pool1 = keras.layers.AveragePooling2D()
        self.conv2 = keras.layers.Conv2D(16, (5, 5), activation='tanh')
        self.pool2 = keras.layers.AveragePooling2D()
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(84, activation='relu')
        self.d2 = keras.layers.Dense(10)

    def call(self, x):
        x = self.preprocess(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

model = LeNet()

# Loss, Optimizer and Metrics
loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()
train_loss = keras.metrics.Mean(name='train_loss')
train_acc = keras.metrics.SparseCategoricalAccuracy(name='train_acc')
test_acc = keras.metrics.SparseCategoricalAccuracy(name='test_acc')
test_loss = keras.metrics.Mean(name='test_loss')

# Training and Testing Step

@tf.function()
def train_step(imgs, labels):
    with tf.GradientTape() as tape:
        logits = model(imgs, training=True)
        losses = loss_object(labels, logits)
    gradients = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(losses)
    train_acc(labels, logits)

@tf.function()
def test_step(imgs, labels):
    logits = model(imgs, training=False)
    losses = loss_object(labels, logits)
    test_loss(losses)
    test_acc(labels, logits)

# The training Loop

EPOCHS = 10

for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()
    train_acc.reset_states()
    test_acc.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    
    for images, labels in test_ds:
        test_step(images, labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_acc.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_acc.result() * 100}'
    )