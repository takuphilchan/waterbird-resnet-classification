# Waterbirds Image Classification

This repository contains a deep learning model for classifying images of waterbirds. The model is built using PyTorch and leverages the Waterbirds dataset (https://drive.google.com/file/d/1xPNYQskEXuPhuqT5Hj4hXPeJa9jh7liL/view) for training and evaluation. It applies computer vision techniques to classify images based on whether the birds are associated with water or land.

## Features
- **Deep Learning Classification**: Implemented using convolutional neural networks (CNNs) and pretrained models.
- **Data Augmentation**: Utilizes various data augmentation techniques to improve model performance.
- **Model Training & Evaluation**: Full pipeline to train, validate, and test the classification model.
- **Logging**: Integrated logging for tracking training progress and model performance.
- **Checkpointing**: Model saving to ensure recovery and the best model is saved based on validation accuracy.

## Installation

### Prerequisites

1. **Python 3.7+**: Make sure you have Python 3.7 or higher installed.
2. **PyTorch**: Install PyTorch with CUDA support (if using GPU).
3. **Other dependencies**: Install the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
