import tensorflow as tf

'''The from_tensor_slices and from_tensors methods
We will start by looking at the from_tensor_slices and the from_tensors methods.

Both static methods are used to create datasets from Tensors or Tensor-like objects, such as numpy arrays or python lists. 
We can also pass in tuples and dicts of arrays or lists. 
The main distinction between the from_tensor_slices function and the from_tensors function is that the from_tensor_slices method 
will interpret the first dimension of the input data as the number of elements in the dataset, whereas the from_tensors method 
always results in a Dataset with a single element, containing the Tensor or tuple of Tensors passed.'''

# We create a random tensor with shape (3,2)
example_tensor = tf.random.uniform([3,2])
print (example_tensor)

# We create 2 different Datasets each with a different method and explore.
dataset1 = tf.data.Dataset.from_tensor_slices(example_tensor)
dataset2 = tf.data.Dataset.from_tensors(example_tensor)
print(dataset1)
print(dataset2)

'''As seen above, creating the Dataset using the from_tensor_slices method 
slices the given array or Tensor along the first dimension to produce a set of elements for the Dataset.

This means that although we could pass any Tensor - or tuple of Tensors - to the from_tensors method, 
the same cannot be said of the from_tensor_slices method, which has the additional requirement that 
each Tensor in the list has the same size in the zeroth dimension.'''

# Lets create 3 tensors with different shapes
tensor1 = tf.random.uniform([10,2,2])
tensor2 = tf.random.uniform([10,1])
tensor3 = tf.random.uniform([9,2,2])

# We cannot create a Dataset from tensor1 and tensor3 since they don't have the same first dimension.
# but we can do it with tensors 1 and 2, the dimension 10 will be interpreted as the "number of elements".
dataset = tf.data.Dataset.from_tensor_slices((tensor1, tensor2))
dataset.element_spec


##### Dataset from Numpy arrays #####
# When done the numpy array is converted into a set of tf.constant operations.

import numpy as np
numpy_array = np.array([[[1,2], [3,4]], [[5,6], [7,8]], [[9,10], [11,12]]])
print(numpy_array.shape)

# Same thing we create two different Datasets with each method.
dataset1 = tf.data.Dataset.from_tensor_slices(numpy_array)
dataset2 = tf.data.Dataset.from_tensors(numpy_array)
print(dataset1.element_spec)
print(dataset2.element_spec)


##### Dataset from a Pandas Dataframe #####

import pandas as pd
pd_dataframe = pd.read_csv('balloon_dataset.csv')
pd_dataframe.head()

# To convert a dataframe we first convert it into a dictionary to preserve the column names.
# Datasets can also be converted from dictionaries where instead of tuples where we acces data by index we will access data by key.

pd_dict = dict(pd_dataframe)
print(pd_dict.keys())

# Let's create the Dataset using from_tensor_slices and notices how when we inspect the element it includes the column names.
pd_dataset = tf.data.Dataset.from_tensor_slices(pd_dict)
pd_dataset.element_spec
next(iter(pd_dataset))


##### Dataset from CSV files #####
# We are going to use something that comes from the "experimental" library which is not officialy released by TF but the community created.
# When using this function we speciy the "target" column which is used to structure the Dataset into a (input, target) tuple.

csv_dataset = tf.data.experimental.make_csv_dataset('balloon_dataset.csv', batch_size=1, label_name='Inflated')
csv_dataset.element_spec
next(iter(csv_dataset))
