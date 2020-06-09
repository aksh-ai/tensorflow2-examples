import cv2
import tensorflow as tf
from layers import *

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_copy = img.copy()
    img = cv2.resize(img, (28, 28))
    img = img/255.0
    img = img.reshape(1, 28, 28, 1)
    return tf.cast(tf.convert_to_tensor(img), tf.float32), cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.saved_model.load('models')

image_path = input('Enter the images path: ')

if image_path:
    image, original = preprocess_image(image_path)

    prediction = model(image)[0]

    print(f"Predicted Digit: {tf.argmax(prediction).numpy()}")

    cv2.putText(original, 'Predicted Digit {}'.format(str(tf.argmax(prediction).numpy())), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 

    cv2.imshow('HDC', original)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

else:
    print('user has not entered a valid path')