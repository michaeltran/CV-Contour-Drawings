import os
import numpy as np

from keras.preprocessing import image
from keras.models import model_from_json

# load json and create model
with open('model/model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("model/model.h5")
    print("Loaded model from disk")

correct_classifications = 0
incorrect_classifications = 0

print('Expecting: dog')
for file_name in os.listdir('data-hacked/dog'):
    test_image = image.load_img('data-hacked/dog/' + file_name, color_mode='grayscale', target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(file_name)
    print(result)
    best_result = np.argmax(result)
    for class_name in training_set.class_indices:
        if best_result == training_set.class_indices[class_name]:
            prediction = class_name
    if prediction == 'dog':
        correct_classifications += 1
    else:
        incorrect_classifications += 1
    print('Prediction: ' + str(best_result) + ' - ' + prediction)
    print('')

print('')
print('')

print('Expecting: human')
for file_name in os.listdir('data-hacked/human'):
    test_image = image.load_img('data-hacked/human/' + file_name, color_mode='grayscale', target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(file_name)
    print(result)
    best_result = np.argmax(result)
    for class_name in training_set.class_indices:
        if best_result == training_set.class_indices[class_name]:
            prediction = class_name
    if prediction == 'human':
        correct_classifications += 1
    else:
        incorrect_classifications += 1
    print('Prediction: ' + str(best_result) + ' - ' + prediction)
    print('')

print("Accuracy: %0.2f" % (correct_classifications / (correct_classifications + incorrect_classifications)))