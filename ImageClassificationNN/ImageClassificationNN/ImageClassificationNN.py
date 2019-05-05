import os
import re
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

class Hacker(object):
    # Original Code that this code is based off of: https://medium.com/@ageitgey/machine-learning-is-fun-part-8-how-to-intentionally-trick-neural-networks-b55da32b7196
    def HackImage(self, original_path, destination_path_hacked, destination_path_original):
        model = inception_v3.InceptionV3()

        # Grab a reference to the first and last layer of the neural net
        model_input_layer = model.layers[0].input
        model_output_layer = model.layers[-1].output

        # Choose an ImageNet object to fake
        # The list of classes is available here: https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
        # Class #859 is "toaster"
        # 151-285 = dogs and cats
        object_type_to_fake = 859

        # Load the image to hack
        img = image.load_img(original_path, target_size=(299, 299))
        original_image = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        original_image /= 255.
        original_image -= 0.5
        original_image *= 2.

        # Add a 4th dimension for batch size (as Keras expects)
        original_image = np.expand_dims(original_image, axis=0)

        # Pre-calculate the maximum change we will allow to the image
        # We'll make sure our hacked image never goes past this so it doesn't look funny.
        # A larger number produces an image faster but risks more distortion.
        max_change_above = original_image + 0.01
        max_change_below = original_image - 0.01

        # Create a copy of the input image to hack on
        hacked_image = np.copy(original_image)
        og_image = np.copy(original_image)

        # How much to update the hacked image in each iteration
        learning_rate = 0.1

        # Define the cost function.
        # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
        cost_function = model_output_layer[0, object_type_to_fake]

        # We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
        # In this case, referring to "model_input_layer" will give us back image we are hacking.
        gradient_function = K.gradients(cost_function, model_input_layer)[0]

        # Create a Keras function that we can call to calculate the current cost and gradient
        grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

        cost = 0.0

        i = 0
        # In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
        # until it gets to at least 80% confidence
        while cost < 0.80:
            i += 1
            # Check how close the image is to our target class and grab the gradients we
            # can use to push it one more step in that direction.
            # Note: It's really important to pass in '0' for the Keras learning mode here!
            # Keras layers behave differently in prediction vs. train modes!
            cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

            # Move the hacked image one step further towards fooling the model
            hacked_image += gradients * learning_rate

            # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
            hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
            hacked_image = np.clip(hacked_image, -1.0, 1.0)

            #if i % 1000 == 0:
            #    Output("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))

        # De-scale the image's pixels from [-1, 1] back to the [0, 255] range
        img = hacked_image[0]
        img /= 2.
        img += 0.5
        img *= 255.

        ## Save the hacked image!
        im = Image.fromarray(img.astype(np.uint8))
        im.save(destination_path_hacked)

        # De-scale the image's pixels from [-1, 1] back to the [0, 255] range
        img = og_image[0]
        img /= 2.
        img += 0.5
        img *= 255.

        ## Save the original image!
        im = Image.fromarray(img.astype(np.uint8))
        im.save(destination_path_original)

        self.PredictImage(destination_path_original)
        self.PredictImage(destination_path_hacked)
        Output()

    def PredictImage(self, path):
        model = inception_v3.InceptionV3()
        img = image.load_img(path, target_size=(299, 299))
        loaded_image = image.img_to_array(img)

        # Scale the image so all pixel intensities are between [-1, 1] as the model expects
        loaded_image /= 255.
        loaded_image -= 0.5
        loaded_image *= 2.

        # Add a 4th dimension for batch size (as Keras expects)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Run the image through the inception_v3 neural network
        predictions = model.predict(loaded_image)

        # Convert the predictions into text and print them
        predicted_classes = inception_v3.decode_predictions(predictions, top=1)
        imagenet_id, name, confidence = predicted_classes[0][0]
        Output(path + " - " + "This is a {} with {:.4}% confidence!".format(name, confidence * 100))

    def HackAllImages(self):
        root_path = '../..'
        picture_path = root_path + '/' + 'data' + '/' + 'photos'
        hacked_path = root_path + '/' + 'data' + '/' + 'hacked_photos'

        for folder_name in os.listdir(picture_path):
            if not os.path.exists(hacked_path + '/' + folder_name):
                os.makedirs(hacked_path + '/' + folder_name)
            for file_name in os.listdir(picture_path + '/' + folder_name):
                name = re.findall("(.+?)(\.[^.]*$|$)", file_name)[0][0]
                self.HackImage(picture_path + '/' + folder_name + '/' + file_name, 
                          hacked_path + '/' + folder_name + '/' + name + '.png',
                          hacked_path + '/' + folder_name + '/' + 'original-' + name + '.png')
        return

def InitializeOutputFile():
    with open('results' + '/' + 'output.txt', 'w') as file:
        file.write('')

def Output(text=''):
    print(text)
    with open('results' + '/' + 'output.txt', 'a') as file:
        file.write(str(text) + '\n')

if __name__ == '__main__':
    InitializeOutputFile()
    hacker_obj = Hacker()
    hacker_obj.HackAllImages()