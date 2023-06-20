# Predicting the Mechanical Characteristics of Corroded Reinforcing Steel through Machine Learning

## 1.	Project Description
This is an example implementation of nine predictive machine learning models used to estimate the yield force capacity of corroded steel bars. The models use an extensive database of 1,349 monotonic tensile tests collected from 26 experimental campaigns available in the literature. The models included in this repository are:

Network-Based Models:

* Artificial Neural Network (ANN).
* Adaptive-Neuro Fuzzy Inference System (ANFIS).


Tree-Based Ensemble Learning Models:
* Gradient Boosting Regression Trees (GBRT).
* Random Forest (RF).
* CatBoost Regression (CBR).


Kernel Machines:
* Gaussian Process Regression (GPR).
* Support Vector Regression (SVR).


Linear Models:
* Generalized Additive Model (GAM).
* Multiple Linear Regression (MLR).


Note that the ANFIS code used in this project is a PyTorch-based implementation sourced from a licensed repository developed by user gregorLen, and can be found at: https://github.com/gregorLen/S-ANFIS-PyTorch. The necessary ANFIS scripts are included in this repository.  

## 2.	Database
The code presented in this repository focuses on the prediction of a single response variable (yield force, *Fy*) and is taken from a larger study predicting several mechanical properties of corroded steel. The article manuscript is currently under review and will be attached once available. Data.csv provides all the necessary data to execute the run.py script. The complete database is available open-source at the DOI: 10.5281/zenodo.8035720.

## 3.	Code Structure
The run.py script can be executed using all other scripts provided in this repository – Functions.py (additional functions) and sanfis.py/helpers.py (ANFIS implementation).

The run.py file is organized in the following format:
* Data Preparation
* Model building and implementation using a 100-fold cross-validation training method.
* Model performance and error evaluation.
* Hyperparameter optimization.

**Note:** Because of the high computational expense of running 100 cross-validation iterations, hyperparameter optimization is not included in the governing for-loop. Example code for hyperparameter optimization is provided at the end of run.py for those models not optimized within the cross-validation training loop.

The run.py script outputs a 9x100 dataframe compiling each model’s performance, evaluated against the R2, MSE, RMSE, MAE, and MAPE. No plotting is included in the current version.

