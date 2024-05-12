
# Plant Disease Classification Model using CNN

[![Coverage]https://img.shields.io/badge/Jupyter%20Notebook-FFA500?style=for-the-badge&labelColor=FFFFFF&logo=jupyter](https://jupyterlab.readthedocs.io/en/stable)



## Overview:

This project aims to classify plant images into different disease categories using Convolutional Neural Networks (CNNs). The dataset used for training and validation consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is obtained from Kaggle, and it is divided into a 80/20 ratio for training and validation sets, while a separate directory containing 33 test images is created for prediction purposes.

## Dataset

1) Source: Kaggle - New Plant Diseases Dataset

2) Description: The dataset is recreated using offline augmentation from the original dataset. It includes images of both healthy and diseased crop leaves, with 38 different disease classes represented.

## Models Used

1) Sequential CNN

2) LeNet-5

3) AlexNet

4) GoogLeNet

5) VGG-16

## Tools and Technologies
1) TensorFlow / Keras: Used for building and training the CNN models.

2) Streamlit: Utilized for hosting the web application, allowing users to upload images from the internet or their desktop for classification.

3) Python: The primary programming language used for model development and application implementation.

## Usage
1) Installation: Ensure Python and required libraries are installed (tensorflow, streamlit, etc.).

2) Training Models: Train the desired model(s) using the provided dataset or customize the training process as needed.

3) Web Application: Run the Streamlit application locally to host the web page for image classification. Users can drag and drop images onto the interface for prediction.

## Limitations
1) Large dataset size may require significant computational resources, particularly GPU, for efficient training.

2) Limited training epochs may impact the model's ability to fully learn intricate patterns in the data.

3) Overfitting and generalization issues may arise, especially with complex datasets and architectures.

4) The model's performance may be constrained by the sequential architecture, potentially limiting its ability to capture complex relationships in the data.

## Future Improvements

1) Experiment with different architectures and hyperparameters to improve model performance.

2) Explore techniques for handling class imbalances within the dataset.

3) Implement advanced regularization techniques to mitigate overfitting.

4) Deploy the application on scalable cloud infrastructure for broader accessibility.

## Contributors
1) Jeevan E G

2) Manoj Y N

3) Nandeesha Kantli

4) Pranav P Kulkarni

## Acknowledgements

1) Credits to the original dataset creators and contributors.

2) Acknowledgement of any external libraries or resources used in the project.

