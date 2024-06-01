# Predicting the Mechanical Characteristics of Corroded Reinforcing Steel through Machine Learning

## 1.	Project Description
This repository includes an example implementation of nine predictive machine learning models used to estimate the yield force capacity of corroded steel bars. The models use an extensive database of 1,349 monotonic tensile tests collected from 26 experimental campaigns available in the literature. The models included in this repository are:

Network-Based Models:

* Artificial Neural Network (ANN).
* Adaptive-Neuro Fuzzy Inference System (ANFIS).

Tree-Based Ensemble Learning Models:

* Gradient Boosting Regression Trees (GBRT).
*	Random Forest (RF).
*	CatBoost Regression (CBR).

Kernel Machines:
*	Gaussian Process Regression (GPR).
*	Support Vector Regression (SVR).
  
Linear Models:

*	Generalized Additive Model (GAM).
*	Multiple Linear Regression (MLR).
  
Note that the ANFIS code used in this project is a PyTorch-based implementation sourced from a licensed repository developed by user gregorLen, and can be found at: https://github.com/gregorLen/S-ANFIS-PyTorch. 
The necessary ANFIS scripts are included in this repository.  
  
## 2.	Database
The code presented in this repository focuses on predicting a single response variable (yield force, $F_{y}$). It is taken from a more extensive study predicting several nine mechanical properties of corroded steel. The article manuscript is currently under review and will be attached once available. Data.csv provides all the necessary data to execute the run.py script. The complete database is available open-source at: https://zenodo.org/records/8035720

## 3. Model Training
Each model is trained and tested using a Repeated k-fold cross-validation approach. A 10-fold split is implemented and repeated ten times, representing a 100-fold approach, with each dataset randomly reshuffled between repetitions.

## 4. Instructions
Follow the instructions below to execute the script and build the models:
1.	Download the zip file containing all files in the repository to your local drive. 
2.	Extract or unzip the folder, keeping all files together without moving the relative path between them. 
3.	Using a Python environment of your choice (e.g., Jupyter Notebook, Visual Studio Code, Spyder, etc.), open the run.py file.
4.	Check that all Python dependencies required to run the script have been installed in your Python environment. See the list below for all the necessary packages. 
5.	Once all the necessary packages are installed, execute the run.py script to train and test the models. 
6.	Note that due to the extensive Repeated K-fold training algorithm adopted in this study, the script will likely take between 2 to 4 hours to run entirely, depending on the CPU of the local device. 


## 5.	Code Structure
The run.py file is organized in the following format:
*	Data Preparation
*	Model building and predictions
*	Model performance and error evaluation.
*	Hyperparameter optimization.


Functions.py includes additional functions required to execute the run.py file. 

sanfis.py/helpers.py is the ANFIS implementation.

Because of the considerable computational expense of running 100 cross-validation iterations for nine different models, hyperparameter optimization has been excluded from the training loop. Example code for hyperparameter optimization is provided at the end of run.py.

The run.py script outputs a 9x100 dataframe compiling each model’s performance, evaluated against the R2, MSE, RMSE, MAE, and MAPE. 

## 6. Related Work

A similar study investigating the residual bending moment of corroded reinforcing beams can be found at: https://github.com/bma114/corroded-RC-beam-moment-capacity

The study also includes an example application allowing users to input new corroded beam data and make predictions using the best-performing models for each response variable. It can be accessed at: https://github.com/bma114/Corroded_Beam_Bending_Prediction 

With the complete open-source database available at: https://zenodo.org/records/8062007 

## 7. Dependencies
The application includes the following dependencies to run:
*	Python == 3.11.0
*	Pandas == 1.4.4
*	NumPy == 1.26.4
*	tqdm == 4.66.4
*	TensorFlow == 2.16.1
*	Keras == 3.3.3
*	Torch == 2.3.0
*	Scikit-Learn == 1.0.2
*	CatBoost == 1.2.5
*	PyGAM == 0.9.1


