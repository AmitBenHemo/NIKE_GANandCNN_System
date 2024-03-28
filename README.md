# Nike Shoes Computer Vision Project

This project is focused on leveraging computer vision techniques to analyze and generate Nike shoes. It is divided into two main parts: classification and generation. The goal is to not only accurately classify different types of Nike shoes but also to creatively generate new shoe designs using deep learning models.

## Part 1: Shoe Classification

### Overview

In the first part of the project, we implemented a Convolutional Neural Network (CNN) to classify Nike shoes into five distinct types. The dataset consists of images of Nike shoes, categorized into these types based on their design, usage, and other distinguishing features.

### Dataset

The dataset includes images of the following five types of Nike shoes:

1. Air Force
2. Air Jordans
3. Air Max
4. Cleats
5. Dunk

Each category contains images that have been manually labeled to ensure accuracy in the training set.

### Model Architecture

The CNN architecture used for this project is designed to extract and learn the most relevant features from shoe images for effective classification. It consists of several convolutional layers, pooling layers, and fully connected layers, optimized for high accuracy on the dataset.

### Training

The model was trained using a split of the dataset into training, validation, and test sets. We employed data augmentation techniques to increase the diversity of the training data, improving the model's robustness and ability to generalize.

### Results

The final model achieved a classification accuracy of XX% on the test set, demonstrating its effectiveness in distinguishing between different types of Nike shoes.

## Part 2: Shoe Generation

### Overview

The second part of the project focuses on generating new types of Nike shoes using a Generative Adversarial Network (GAN). The aim was to explore creative designs that could potentially inspire future Nike shoe products.

### Dataset

The same dataset used in the classification part was also utilized for training the GAN, providing a diverse range of shoe images for learning.

### Model Architecture

The GAN architecture consists of two main components: the Generator and the Discriminator. The Generator is responsible for creating new shoe images, while the Discriminator evaluates their authenticity compared to real images from the dataset.

### Training

The GAN was trained until it reached a point where the generated shoe images were indistinguishable from real images to the Discriminator. This process involved careful tuning of parameters and monitoring of the training progress to ensure a balance between the Generator and Discriminator.

### Results

The trained GAN was able to generate a variety of new and unique Nike shoe designs, showcasing the potential of GANs in creative design processes.

## Technologies Used

- Python
- TensorFlow / Keras
- NVIDIA CUDA (for GPU acceleration)

## Project Structure

```
nike-shoes-cv-project/
│
├── classification/
│   ├── dataset/
│   ├── models/
│   └── train.py
│
└── generation/
    ├── dataset/
    ├── models/
    └── train.py
```

## Setup and Usage

Instructions on how to set up the project environment, train the models, and generate new shoe designs can be found in the respective directories for classification and generation.

## License

This project is open-sourced under the MIT license. See the LICENSE file for more details.

## Acknowledgments

This project was created for educational purposes and is not affiliated with Nike, Inc. The dataset was compiled from publicly available images of Nike shoes.
