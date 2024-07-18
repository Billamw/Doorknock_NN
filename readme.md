# Door Knock Detection Project

## Overview

This project focuses on developing a Convolutional Neural Network (CNN) for the purpose of detecting knock sounds within audio recordings. The model is designed to differentiate between knock sounds and background noise, leveraging the power of deep learning to achieve high accuracy in sound classification.

## Data Sources

The training and validation datasets consist of a combination of own recorded audio samples and publicly available datasets. Specifically, the project utilizes:

-   **Own Recorded Audios**: Custom recordings of knock sounds and various background noises, captured in different environments to ensure the robustness of the model.
-   [**Soundata**](https://github.com/soundata/soundata): A comprehensive collection of sound datasets, used to augment the variety of knock and noise sounds in the training data.
    -   **FSD50K**: A publicly available dataset containing a wide range of sound classes, including knock sounds, which has been instrumental in training the model to recognize a variety of knock characteristics.

## Model Architecture

The CNN architecture for this project is inspired by the guide available on [Towards Data Science](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5). This guide provides a step-by-step approach to sound classification using deep learning, which has been adapted and optimized for the specific task of knock detection in this project.

## Implementation

The model is implemented using PyTorch, a leading deep learning library that offers flexibility and a wide array of tools for model development and training.
