# Rock Paper Scissors Classifier

This project is a simple image classifier that predicts whether an uploaded image contains rock, paper, or scissors. It is the result of my documentation during my participation in the "Dicoding Bootcamp."

## Components

- **Dataset Folder:** Contains the training and validation images for the classifier. The dataset is organized into separate folders for each class (rock, paper, scissors).

- **model.ipynb:** Jupyter Notebook containing the code for training the machine learning model. This notebook loads the dataset, preprocesses the images, defines the model architecture, trains the model, and evaluates its performance.

- **app.py:** Python script for the web application that allows users to upload an image and obtain predictions from the trained model. The web app is built using Streamlit.

- **model.h5:** The trained machine learning model saved in HDF5 format. This file contains the learned parameters and architecture of the model after training.

## Usage

To use the classifier, follow these steps:

1. Ensure you have Python and the required dependencies installed.
2. Clone or download this repository to your local machine.
3. Navigate to the project directory in your terminal.
4. Train the model by running `model.ipynb` in a Jupyter Notebook environment.
5. Once trained, run the web application by executing `streamlit run app.py` in your terminal.
6. Upload an image containing rock, paper, or scissors to the web app.
7. Click the "Predict" button to obtain predictions from the trained model.

## Dependencies

- Python 3.11
- TensorFlow
- Keras
- Streamlit
