import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19

vgg_model = VGG19()

vgg_input = vgg_model.input
vgg_layers = vgg_model.layers
vgg_model.summary()

# Retrieving the output layers of the model and plotting
layer_outputs = [layer.output for layer in vgg_layers]
features = Model(inputs=vgg_input, outputs=layer_outputs)

#Plot both models (they are the same as you can see)
tf.keras.utils.plot_model(vgg_model, 'VGG19.png', show_shapes=True)
tf.keras.utils.plot_model(features, 'features_VGG19.png', show_shapes=True)

img = np.random.random((1,244,244,3)).astype('float32')
extracted_features = features(img)

import IPython.display as display
from PIL import Image

#display.display(Image.open('cool_cat.jpg'))

######### Now we can extract the features of any layer of the image passing through the model indexing de layer we'd like to see
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image

imgpath = 'cool_cat.jpg'
img = image.load_img(imgpath, target_size=(224,244))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

extracted_features = features(x)
f2 = extracted_features[3]
print (f2.shape)

imgs = f2[0,:,:]
plt.figure(figsize=(15,15))
for n in range(16):
    ax = plt.subplot(4,4,n+1)
    plt.imshow(imgs[:,:,n])
    plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# We can do it also by indexing using the name of the layer
extracted_features_block5_conv4 = Model(inputs=features.input, outputs=features.get_layer('block5_conv4').output)
block5_conv4_features = extracted_features_block5_conv4.predict(x)

imgs = f2[0,:,:]
plt.figure(figsize=(15,15))
for n in range(16):
    ax = plt.subplot(4,4,n+1)
    plt.imshow(imgs[:,:,n])
    plt.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.01)