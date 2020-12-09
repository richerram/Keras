import numpy as np
import tensorflow as tf

strings = tf.Variable (["Hello World!"], tf.string)
floats = tf.Variable ([3.14159, 2.71828], tf.float64)
ints = tf.Variable ([767,], tf.int32)
complexs = tf.Variable ([25.9 - 12.4j], tf.complex128)

tf.Variable(tf.constant(4.2, shape=(3,3)))

v = tf.Variable(0.0)
w = v+1
print (type(w))

v.assign_add(1)
print(v)

v.assign_sub(1)
print(v)

x = tf.constant([[3,6,5], [9,3,7], [4,9,2]])
print(x)
print ('dtype', x.dtype)
print ('shape', x.shape)

x.numpy()
print (x)

x = tf.constant([[3,6,5], [9,3,7], [4,9,2]], dtype=tf.float32)
print (x.dtype)

coeffs = np.arange(16)
print (coeffs)
shape1 = [4,4]
shape2 = [8,2]
shape3 = [2,2,2,2]
a = tf.constant(coeffs, shape=shape1)
print ('\n a: \n',a)
b = tf.constant(coeffs, shape=shape2)
print ('\n a: \n',b)
c = tf.constant(coeffs, shape=shape3)
print ('\n a: \n',c)

t = tf.constant(np.arange(80), shape=[5,2,8])
rank = tf.rank(t)
print ('rank: ',rank)
t2 = tf.reshape(t, [8,10])
print ('t2.shape: ', t2.shape)

ones = tf.ones(shape=(2,3))
print ('ones: ',ones)
zeros = tf.zeros(shape=(3,2))
print ('zeros: ',zeros)
eye = tf.eye(3)
print ('eye: ', eye)
tensor7 = tf.constant(7.0, shape=[2,3])
print ('tensor7: ', tensor7)

t1 = tf.ones(shape=(2,2))
t2 = tf.zeros(shape=(2,2))
concat0 = tf.concat([t1,t2], 0)
print ('concat0: ', concat0)
concat1 = tf.concat([t1,t2], 1)
print('concat1: ', concat1)

#Expanding a tensor (like adding an axis on NumPy...np.newaxis...)
t = tf.constant(np.arange(24), shape=(3,2,4))
t1 = tf.expand_dims(t,0)
print(f't1.shape: {t1.shape}\n t1: {t1}')
t2 = tf.expand_dims(t,1)
print(f't2.shape: {t2.shape}\n t1: {t2}')
t3 = tf.expand_dims(t,3)
print(f't3.shape: {t3.shape}\n t1: {t3}')

#Sqeezing a tensor
t1 = tf.squeeze(t1,0)
print (f't1.shape back to normal: {t1.shape}')
t2 = tf.squeeze(t2,1)
print (f't2.shape back to normal: {t2.shape}')
t3 = tf.squeeze(t3,3)
print (f't3.shape back to normal: {t3.shape}')

#Slicing a tensor
x = tf.constant([1,2,3,4,5,6,7])
print(x[1:4])

#Doing MATH
c = tf.constant([[1.0,2.0],[3.0,4.0]])
d = tf.constant([[1.0,1.0], [0.0,1.0]])

matmul_cd = tf.matmul(c,d)
print(f'matmul: {matmul_cd}')

c_times_d = c*d
c_plus_d = c+d
c_minus_d = c-d
c_div_d = c/d
print(f'c*d: {c_times_d}\nc+d: {c_plus_d}\nc-d: {c_minus_d}\nc/d: {c_div_d}')

a = tf.constant([[2,3], [3,5]])
b = tf.constant([[2,3], [4,3]])
x = tf.constant([[-6.89 + 1.78j], [-2.54 + 2.15j]])

absx = tf.abs(x)
print (f'absx: {absx}')

powab = tf.pow(a,b)
print(f'powab: {powab}')

#Normal Distribution tensor
tn = tf.random.normal(shape=(2,2), mean=0, stddev=1.)
print (f'Normal Distribution: {tn}')

#Uiform Distribution tensor
tu = tf.random.uniform(shape=(2,2), minval=0, maxval=10, dtype='int32')
print(f'Uniform Distribution: {tu}')

#Poisson Distribution tensor
tp = tf.random.poisson((2,2), 5)
print(f'Poisson Distribution: {tp}')
