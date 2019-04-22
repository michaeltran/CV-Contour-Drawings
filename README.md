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

## How to Run

### Data

Run the following command: `python ImageClassificationNN/ImageClassificationNN/TrainNeuralNetworkSplit.py`

### Train Neural Network

Run the following command: `python ImageClassificationNN/ImageClassificationNN/TrainNeuralNetwork.py`