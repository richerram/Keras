from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50()
print (model.summary())

img_input = image.load_img ('me.jpg', target_size=(224,224))
img_input = image.img_to_array (img_input)
img_input = preprocess_input(img_input[np.newaxis,...])

preds = model.predict(img_input)
decoded_pred = decode_predictions(preds, top=3)[0]
print (decoded_pred)