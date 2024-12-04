# test the model on your own images
# the example is prepared for real / ai-generated faces detection, but you can load any model with any categories

import numpy as np
from almus.train import CNN
from tensorflow.keras.preprocessing.image import load_img


def preprocess_image(image_path, image_size):
    target_size = image_size
    image = load_img(image_path, target_size=target_size)
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict_image(model, img_path, image_size, categories):
    # Image preprocessing
    image = preprocess_image(image_path=img_path, image_size=image_size)
    image_data = np.expand_dims(np.array(image) / 255.0, axis=0)
    image_data = image_data.squeeze(0) # to eliminate one unnecessary dimension (lacks testing for multiple models, datasets and images)

    # Model prediction
    prediction = model.predict(image_data)
    predicted_label = categories[np.argmax(prediction)]
    print(predicted_label)


# path to your model
cur_model = CNN(input_shape=(300, 300, 3),
                num_classes=2,
                model='../models/real_and_fake.h5')

# paths to your images (change to your own)
images = ['C:/Users/Lenovo/Downloads/Telegram Desktop/photo_2024-12-02_16-29-32.jpg',
          'C:/Users/Lenovo/Downloads/Telegram Desktop/photo_2024-12-02_16-35-30.jpg']

# categories for chosen dataset (please ensure the categories order matches the categories order in train and test sets)
categories = ['fake', 'real']

# predict all the images
for image in images:
    predict_image(model=cur_model, img_path=image, image_size=(300, 300), categories=categories)



