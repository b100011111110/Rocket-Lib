import tensorflow as tf
import numpy as np

x = tf.convert_to_tensor(np.ones((10, 5), dtype=np.float32))
layer = tf.keras.layers.ActivityRegularization(l2=0.001)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = layer(x)
    loss = sum(layer.losses)
    print("Layer losses:", layer.losses)

grad = tape.gradient(loss, x)
print("Gradient:", grad[0, 0].numpy())
