import tensorflow as tf

var = tf.Variable([1.0])
grad = tf.constant([0.1])
opt = tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-7)

opt.apply_gradients([(grad, var)])
print("After 1 step:")
print("Var:", var.numpy())
print("m:", opt.get_slot(var, "m").numpy())
print("v:", opt.get_slot(var, "v").numpy())
