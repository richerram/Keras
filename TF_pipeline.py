##### Automatic Differentiation example #####
import tensorflow as tf

x= tf.constant(2.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2
    grad = tape.gradient(y, x)
print(grad)
# tf.Tensor(4.0, shape=(), dtype=float32)


#### We can also take the gradient with respect to a medium tensor like "y" here.
x = tf.constant([0, 1, 2, 3], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x**2)
    z = tf.math.sin(y)
    dz_dy = tape.gradient(z, y)
print(dz_dy)
# tf.Tensor(0.13673723, shape=(), dtype=float32)


#### We can also take it with respect to Multiple Tensors!!!
x = tf.constant([0,1,2,3], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.reduce_sum(x**2)
    z = tf.math.sin(y)
    dz_dy, dz_dx = tape.gradient(z, [y, x])
print(dz_dy)
# tf.Tensor(0.13673723, shape=(), dtype=float32)
print(dz_dx)
# tf.Tensor([0.         0.27347445 0.5469489  0.82042336], shape=(4,), dtype=float32)