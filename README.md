ANN vs CNN on CIFAR-10 (Hugging Face)

This project compares a fully connected Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) on the CIFAR-10 image classification dataset using PyTorch and Hugging Face datasets.
The goal is not just accuracy — but to understand why CNNs outperform ANNs on image tasks.

Problem Statement
Fully connected networks (ANNs) treat images as flat vectors, ignoring spatial structure.

Convolutional Neural Networks (CNNs) introduce inductive bias through:
* Local receptive fields
* Weight sharing
* Translation invariance

This project investigates:
* How performance differs
* How parameter count differs
* How training dynamics differ
* When architecture choice matters

Dataset
CIFAR-10 (via Hugging Face)
* 60,000 32x32 RGB images
* 10 classes
* 50,000 training / 10,000 test

Model Architectures

🔹 ANN (Baseline)
* Input: Flattened 32×32×3 image (3072 features)
* Fully connected layers
* ReLU activations
* Dropout
* Output: 10-class softmax
Limitation: Ignores spatial structure.

🔹 CNN
* Convolutional blocks
* Max pooling
* Feature maps
* Fully connected classifier head
Strength: Learns hierarchical spatial features.

