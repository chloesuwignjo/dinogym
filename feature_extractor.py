from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor: 
    def __init__(self):
        # pre-trained VGG16 model
        base_model = VGG16(weights="imagenet")
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    def extract(self,img):
        img = img.resize((224, 224))
        img = img.convert("RGB")
        # convert to np.array
        img_array = image.img_to_array(img)
        # add an extra dimension to match the input shape expected
        img_array = np.expand_dims(img_array, axis = 0)
        # preprocess the input image to match the format required
        img_array = preprocess_input(img_array)
        # extract features from input image
        feature = self.model.predict(img_array)[0]
        # normalize
        return feature/np.linalg.norm(feature)
    