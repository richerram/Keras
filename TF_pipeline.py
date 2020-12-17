import tensroflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
####IMPORT GENERATOR
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#####Generator OBJECT###
# 'rescale' means all pixels will be rescaled to be between 0 and 1.
# 'horizontal_flip' means that it randomly will flip the images so we are already increasing the amount of exmaples we have.
# 'height_shift_range' means it will randomly move the images 20% up or down and then we will need 'fill_mode'.
# 'fill_mode' will complete the shifted images with some pixels, in this case we pick the "nearest" pixels like: aaa\abc\ccc
# 'featurewise_center' means it will make the "mean" of each feature (let's say the R, G and B layers of an image) each one equal to "0".
image_gen = ImageDataGenerator(rescale=1/255., horizontal_flip=True, height_shift_range=0.2, fill_mode='nearest', featurewise_center=True)

# IMPORTANT #
# Once we have created the generator we have to "fit" it to the training data first and before anything else so it can calculate the features.
image_gen = image_gen.fit(x_train)

# FINALLY - get the generator itself using the flow method.
train_datagen = image_gen.flow(x_train, y_train, batch_size=16)

# Ready to train
model.fit_generator(train_datagen, epochs=20)
