# Computer Vision - Contour Drawings

Evaluation Project based on [PhotoSketch](https://github.com/mtli/PhotoSketch)

## Project Goals

1. Generate image contours from PhotoSketch and deal the images into Train/Test/Validation splits.

2. Train a Convolutional Neural Network (CNN) based on the image contours generated from PhotoSketch.

3. Generate adversarial examples of the original images that fool InceptionV3 into misclassifying.

4. Regenerate image contours using PhotoSketch from adversarial examples.

5. Test the image contours against our trained CNN from step 2.

6. Gather our findings and results.

---

## Requirements

```Rich Text Header
Python 3.6.8
Anaconda
```

```Rich Text Header
Tensorflow
CUDA Toolkit 10.1
cuDNN
h5py
Keras
```

---

## Credits

Images from: [https://www.pexels.com](https://www.pexels.com)

---

## How to Run - Hacked Image Generation

### Create Hacked Images

Run the following command: `python ImageClassificationNN/ImageClassificationNN/ImageClassificationNN.py`.

This will generate all hacked images from the images located in the folder `data/photos` and create a new folder `data/hacked_photos`.

---

## How to Run - PhotoSketch

### Download Pretrained Model

Download the Pretrained model of PhotoSketch from their website. Then place it in `PhotoSketch/Exp/PhotoSketch/Checkpoints/pretrained`.

### Create Image Contours

To generate contours based on original images, edit the command file `scripts/test_pretrained_dogs.cmd` and `scripts/test_pretrained_humans.cmd` (change the dataDir to the location of the PhotoSketch folder).

Then run the scripts from the PhotoSketch root `scripts/test_pretrained_dogs.cmd` and `scripts/test_pretrained_humans.cmd`. The results can be found in `PhotoSketch/Exp/Photosketch/Results`.

Edit and run `"scripts/test_pretrained_dogs_hacked.cmd"` and `"scripts/test_pretrained_humans_hacked.cmd"` just like the steps above to create the contours for the hacked images.

These generated contours should be manually moved into the `presplit/class` folder for neural network processing.

---

## How to Run - Neural Networks

### Split Data

Run the following command: `python ImageClassificationNN/ImageClassificationNN/TrainNeuralNetworkSplit.py`

This will split the data in the `presplit` folder into `test`/`train`/`validation` folder splits.

### Train Neural Network

Run the following command: `python ImageClassificationNN/ImageClassificationNN/TrainNeuralNetwork.py`

This will train the neural network on the data in `train` folder. It will then test it against the `test` folder. Final accuracies will be based on the `validation` folder.