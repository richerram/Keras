import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

a = Input(shape=(128,128,3), name='input_a')
b = Input(shape=(64,64,3), name='input_b')

conv = Conv2D(32,(6,6), padding="SAME")

conv_out_a = conv(a)
print(f'\nConv_Out_a:\n {conv_out_a}')

print(f'Conv Input:\n{conv.input}\n')
print(f'Conv Output:\n{conv.output}\n')

# Creating a new layer node
conv_out_b = conv(b)
print(f'Conv Input:\n{conv.input}\n')
print(f'Conv Output:\n{conv.output}\n')

# print(conv.input_shape) --------- you get an error here since now there are different inputs with different shapes.
# print(conv.output_shape) ---------- same here. ERROR.

# We have to INDEX using 'get_input_shape_at'
print(conv.get_input_shape_at(0))
print(conv.get_input_shape_at(1))
print(conv.get_output_at(0).name)
print(conv.get_output_shape_at(0))
print(conv.get_output_at(1).name)
print(conv.get_output_shape_at(1))


