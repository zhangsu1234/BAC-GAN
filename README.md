BAC-GAN: Missing Data Recovery 
Project Introduction
This project proposes a missing data recovery method (BAC-GAN) for power systems based on an improved Generative Adversarial Network (GAN). Integrating Bidirectional Long Short-Term Memory (BiLSTM), Multi-Head Attention mechanism, and Convolutional Neural Network (CNN), this method can effectively capture the temporal dependencies and local features in power data, enabling high-precision missing data imputation.

Model Structure
Generator: Utilizes BiLSTM and Multi-Head Attention mechanism to enhance the modeling capability for time-series data.
Discriminator: Adopts a CNN structure and incorporates the Hint mechanism to improve the ability to distinguish generated data.

📊Datasets

Two public datasets are used for experiments in this project:
Residential Load Dataset (RLD)
Contains residential electricity consumption data, suitable for analyzing household electricity usage behavior.
Number of samples: 158
Number of features: 96
Smart Meter Dataset (SMD)
Contains hourly electricity consumption data (kWh) from smart meters.
Number of samples: 8760
Number of features: 134

🧪Comparative Methods

To evaluate the performance of BAC-GAN, five mainstream missing data imputation methods are selected for comparative experiments:

KNN (K-Nearest Neighbors): A traditional statistical method that fills missing values using the average of k nearest neighbor samples.
VAE (Variational Autoencoder): A deep learning method based on probabilistic modeling, which learns data distribution through encoding-decoding structures.
GAIN (GAN-based Imputation Method): A classic GAN-based imputation method, which uses a discriminator with a Mask mechanism to optimize data generation.
M-RNN (Recurrent Neural Network-based Imputation Method): A sequence modeling method that captures temporal dependencies using RNN.
MIVAE (Multimodal Variational Autoencoder): An improved VAE method for multimodal data, enhancing adaptability to complex data structures.


🛠️Environmental Dependencies
Ensure the following software and libraries are installed before running the model:

Programming Language: Python 3.8 or higher
Deep Learning Framework: PyTorch 2.0.0 or higher
GPU Acceleration: CUDA 11.8 (recommended for accelerating model training; CPU training is supported but may take longer)
Auxiliary Libraries:
numpy (for numerical computation)
pandas (for data preprocessing)
matplotlib/seaborn (for result visualization)
scikit-learn (for data normalization and evaluation metric calculation)
jupyter (for running notebook files)

🚀Install dependencies via pip (example command):

bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
Run the Main Model (BAC-GAN)
Follow the steps below to start the BAC-GAN model for missing data imputation:

Clone or Download the Project: Ensure all model files (including BAC-GAN.ipynb, data preprocessing scripts, and model definition modules) are in the same directory.
Launch Jupyter Notebook:
Open the terminal, navigate to the project root directory, and execute the following command to start the Jupyter Notebook service:
bash
jupyter notebook

Open and Run the BAC-GAN Notebook:
In the Jupyter Notebook interface displayed in the browser, find and click BAC-GAN.ipynb to open the model execution file.
Follow the step-by-step instructions in the notebook:
Load and preprocess the dataset (e.g., normalize data, simulate missing values).
Initialize the Generator and Discriminator of BAC-GAN.
Set training parameters (e.g., learning rate, number of epochs, batch size).
Start model training and monitor the loss curve.
Perform missing data imputation using the trained model and evaluate results (e.g., calculate MAE, RMSE).
