# Dog-Breed-Classification-Pytorch

## Overview

This repository contains code for a deep learning project aimed at identifying dog breeds from images. The model is trained on a dataset consisting of images of various dog breeds.

## Dataset

The dataset used for training the model is sourced from [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) competition on Kaggle. It contains 10,222 images of 120 different dog breeds. Each image is labeled with the corresponding breed.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- OpenCV
- tqdm
- Matplotlib

## Model Architecture

The model architecture used for this project is ResNet-50, a deep convolutional neural network pre-trained on the ImageNet dataset. The fully connected layer of the model is replaced with a new one to classify dog breeds.

## Results

After training for 20 epochs, the model achieved an accuracy of approximately 78% on the validation set.

## License

This project is licensed under the [MIT License](LICENSE).
