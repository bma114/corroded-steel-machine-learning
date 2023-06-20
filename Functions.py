import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.preprocessing import OrdinalEncoder


##############################################################################################################################


def load_data() -> pd.DataFrame: # load csv file
    return pd.read_excel("Data.xlsx", keep_default_na=False)
#     return pd.read_excel("Data.xlsx", keep_default_na=False, na_values=['_'])
    # keep_default_na=False ensures data with N/A are interpretted as strings not missing numbers.

    
def col_type(df) -> pd.DataFrame: # Seperate features into different types 
    
    # Define lists of column labels by type - categorical, numerical, response (output)
    cat_cols = ["Specimen_Type", "Load_History", "Bar_Type", "Steel_Origin", "Bar_Grade", 
                "Corrosion_Method", "Cathode_Type", "Cleaning_Method"] # Categorical Columns

    num_cols = ["Nominal Diameter", "Gauge Length", "Current Density", "Exposure Duration", "Solution Concentration", 
                "Mass Loss", "Corroded Length"] # Numerical Columns
    
    # ANFIS model only takes a maximum of 8 input features - trim the input lists.
    anfis_cols = ["Specimen_Type", "Load_History", "Bar_Grade", "Corrosion_Method", "Nominal Diameter", 
                  "Current Density", "Exposure Duration", "Mass Loss"] 

    out_col = ["ln(Fy)"] # Output Column [target]
    
    return cat_cols, num_cols, anfis_cols, out_col
    
    
def encoder(df): # Encode categorical variables
    enc = OrdinalEncoder()
    df = enc.fit_transform(df)
    return df


def feature_scaling(df): # Scale all features
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


# Define ANFIS model parameters
def fis_params():
    # Model only works with maximum of 8 input variables.
    # Set 8 input variables with 3 gaussian membership functions each.
    MEMBFUNCS = [
        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}},

        {'function': 'gaussian', 'n_memb': 3, 
         'params': {'mu': {'value': [-0.5, 0.0, 0.5], 'trainable': True}, 
         'sigma': {'value': [1.0, 1.0, 1.0], 'trainable': True}}}]

    # Model hyperparameters
    param = {"n_input": 8, "n_memb": 8,"batch_size": 25, 
             "memb_func": 'gaussian', "scaler": 'Std', 
             "n_epochs": 100, "lr": 0.005, "patience": 100, 
             "delta": 1e-6, "sigma": 0.1}
    
    return MEMBFUNCS, param



# Error Functions
def r_squared(Y, y_hat):
    y_bar = Y.mean()
    ss_res = ((Y - y_hat)**2).sum()
    ss_tot = ((Y - y_bar)**2).sum()
    return 1 - (ss_res/ss_tot)

def mean_squared_err(Y, y_hat):
    var = ((Y - y_hat)**2).sum()
    n = len(Y)
    return var/n

def root_mean_squared_err(Y, y_hat):
    MSE = mean_squared_err(Y, y_hat)
    return np.sqrt(MSE)

def mean_abs_err(Y, y_hat):
    abs_var = (np.abs(Y - y_hat)).sum()
    n = len(Y)
    return abs_var/n

def mean_abs_perc_err(Y, y_hat):
    mape = np.mean(np.abs((Y - y_hat)/ y_hat))*100
    return mape







