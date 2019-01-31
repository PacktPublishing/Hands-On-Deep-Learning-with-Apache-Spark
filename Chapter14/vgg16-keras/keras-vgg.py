from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
from keras.applications.vgg16 import decode_predictions

model = VGG16(weights='imagenet', include_top=True)
print(model.summary())

img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
for i in range(len(features)):
        print("Predicted=%s" % (features[i]))

# convert the probabilities to class labels
#label = decode_predictions(features)
# retrieve the most likely result, e.g. highest probability
#label = label[0][0]
# print the classification
#print('%s (%.2f%%)' % (label[1], label[2]*100))

model_json = model.to_json()
with open('vgg-16.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights("vgg-16.h5")
