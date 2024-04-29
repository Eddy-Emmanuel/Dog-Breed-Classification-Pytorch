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

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/dog-breed-identification.git
   ```

2. Navigate to the project directory:

   ```bash
   cd dog-breed-identification
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/c/dog-breed-identification) and place it in the `data` directory.

5. Preprocess the data and train the model by running:

   ```bash
   python train.py
   ```

6. Evaluate the trained model:

   ```bash
   python evaluate.py
   ```

## Model Architecture

The model architecture used for this project is ResNet-50, a deep convolutional neural network pre-trained on the ImageNet dataset. The fully connected layer of the model is replaced with a new one to classify dog breeds.

## Results

After training for 20 epochs, the model achieved an accuracy of approximately 78% on the validation set.

## License

This project is licensed under the [MIT License](LICENSE).
