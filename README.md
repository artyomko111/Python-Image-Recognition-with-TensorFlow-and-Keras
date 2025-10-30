# CIFAR-10 Image Classification with Convolutional Neural Networks (CNN)

This repository contains the source code and documentation for a final coursework project focused on image recognition using deep learning techniques.

## 🎯 Project Overview

The primary goal of this project was to implement, train, and evaluate a custom Convolutional Neural Network (CNN) for multi-class image classification. The core task involves successfully classifying images from a popular benchmark dataset.

* **Topic:** Python Image Recognition with TensorFlow and Keras
* **Model Architecture:** Custom CNN built sequentially.
* **Core Libraries:** TensorFlow and Keras
* **Programming Language:** Python
* **Dataset:** CIFAR-10 (comprising 60,000 $32\times32$ color images across 10 classes, e.g., airplanes, dogs, cats, cars, etc.)

## 📊 Key Results

The model was trained for **25 epochs** using the Adam optimizer.

| Metric | Value |
| :--- | :--- |
| **Final Test Accuracy** | **83.01%** |
| **Epochs Trained** | 25 |
| **Total Parameters** | 2,264,458 |

## ⚙️ Installation and Setup

### Prerequisites

You need Python 3.x installed on your system.

### Dependencies

Install the required libraries using `pip`. You can create a `requirements.txt` file listing:

```
bash
tensorflow
keras
numpy
```
Then run:
```
Bash

pip install -r requirements.txt
```
### Repository Cloning

Clone the project to your local machine:
```
Bash

git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/[YOUR_REPO_NAME].git
```
Navigate to the project directory:
```
Bash
cd [YOUR_REPO_NAME]
```
## 🚀 Usage

To train and evaluate the CNN model, run the main script:
```
Bash

python cnn_cifar10_classifier.py
```
(Note: Replace cnn_cifar10_classifier.py with the actual name of your Python file.)

The script will automatically download the CIFAR-10 dataset, preprocess the images, build the model, and begin training, printing the loss and accuracy metrics at the end of each epoch.

## 🧠 Model Architecture Summary

The CNN utilizes several key components of a modern deep learning model:

    Convolutional Layers (Conv2D): Used for feature extraction.

    Activation Function (ReLU and Softmax): ReLU adds non-linearity, and Softmax provides the final probability distribution for classification.

    Pooling Layers (MaxPooling2D): Reduces dimensionality and helps prevent overfitting.

    Flatten Layer: Converts the 2D feature maps into a 1D vector for input to the fully connected layers.

    Fully Connected Layers (Dense): Performs the final classification based on the extracted features.

## 📚 References
1. Creating a Python image classifier using TensorFlow 2 and Keras. (n.d.). Waksoft. [cite_start]Retrieved from: https://waksoft.susu.ru/2021/04/03/kak-sozdat-klassifikator-izobrazhenij-na-python-s-pomoshhyu-tensorflow-2-i-keras/ [cite: 198]
2. Python Image Recognition with TensorFlow and Keras. (n.d.). Evileg. [cite_start]Retrieved from: https://evileg.com/ru/post/619/ [cite: 199]
3. Image recognition using AI (TensorFlow). (n.d.). Python Book. [cite_start]Retrieved from: https://pythonbook.site/python/image-recognition-tensorflow?ysclid=liafkpa9ct378876087 [cite: 200]
