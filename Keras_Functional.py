import tensorflow as tf

# Check all devices
print (f'All Devices:\n{tf.config.list_physical_devices()}\n')

#Check GPU devices
print('GPU Devices:\n{}\n'.format(tf.config.list_physical_devices('GPU')))

#Check CPU devices
print('CPU Devices:\n{}\n'.format(tf.config.list_physical_devices('CPU')))

#Get devices names
print(f'Name of the GPU device: {tf.test.gpu_device_name()}\n')

# We can specify which tensor runs on which device, first we will check where it is placed automatically.
x = tf.random.uniform([3,3])
print (f'Device where X tensor is on: {x.device}\n')

print("Is the Tensor on CPU #0:  "),
print(x.device.endswith('CPU:0'))
print('')
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

# Here we can place an operatino manually and check processing times.
import time

def time_add(x):
    start = time.time()
    for loop in range(10):
        tf.add(x,x)
    result = time.time()-start
    print ('Matrix addition (10 loops): {:0.2f}ms'.format(1000*result))

def time_mul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x,x)
    result = time.time() - start
    print("Matrix multiplication (10 loops): {:0.2f}ms".format(1000 * result))

# Force execution on CPU
print ('On CPU:')
with tf.device('CPU:0'):
    x = tf.random.uniform([1000,1000])
    assert x.device.endswith('CPU:0')
    time_add(x)
    time_mul(x)

# Force execution on GPU
print ('On GPU:')
with tf.device('GPU:0'):
    x = tf.random.uniform([1000,1000])
    assert x.device.endswith('GPU:0')
    time_add(x)
    time_mul(x)