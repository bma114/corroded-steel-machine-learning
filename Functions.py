# Import packages
import pandas as pd
import numpy as np

# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import catboost as cb # CatBoost package
from pygam import LinearGAM, s, f # pyGAM package for linear regression GAMs



### ===================================================== ###



# Load the data from csv file
def load_data() -> pd.DataFrame:
    
    return pd.read_csv("Data.csv", keep_default_na=False)

    

# Define the different feature types
def col_type(df) -> pd.DataFrame:
    
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
    


# Encode categorical variables   
def encoder(df): 
    enc = OrdinalEncoder()
    df = enc.fit_transform(df)
    return df



# Normalize all features
def feature_scaling(df): 
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



# Define error metrics
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



# Build ANN architecture
def ann_architecture():

    ann_model = Sequential()
    ann_model.add(Dense(128, input_dim=15, kernel_initializer='normal', activation='relu'))
    ann_model.add(Dense(64, kernel_initializer='normal', activation='sigmoid'))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])  

    return ann_model



# Build GBRT model
def build_gbrt(): # Add optimized hyperparameters into GBRT model

    gbrt_model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.2, max_depth=2, 
                                            max_leaf_nodes=5, min_samples_leaf=1, min_samples_split=8, 
                                            random_state=0, loss='squared_error')            
    return gbrt_model



# Build CBR model
def build_cb():

    # Build model
    cb_model = cb.CatBoostRegressor(
        loss_function='MAE', 
        iterations=400, 
        learning_rate=0.1, 
        depth=4, 
        l2_leaf_reg=0.5, 
        verbose=False, 
        thread_count=-1)
    
    return cb_model



# Build RF model
def build_rf():

    rf_model = RandomForestRegressor(n_estimators=400, max_depth=3, max_features=1.0, 
                                        min_samples_leaf=1, min_samples_split=2, random_state=25, 
                                        bootstrap=True, n_jobs=-1, criterion='squared_error')
    return rf_model



# Build SVR model
def build_svr():

    svr_model = SVR(kernel='rbf', gamma='scale', C=10)

    return svr_model



# Build GPR model
def build_gpr():

    kernel = ConstantKernel(1.0, (1e-1, 1e3)) * RBF(10, (1e-3, 1e3))

    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

    return gpr_model



# Build GAM model
def build_gam(X_train, Y_train):

    # Define gridsearch parameter ranges
    grid_splines = np.linspace(10,30,20) # number of splines per feature
    lams = np.random.rand(40, 15) # lambda value for smoothing penalization
    lams = lams * 15 - 3 # Search space for lam needs 15 dimensions for a model with 15 lam terms (one per feature)
    lams = np.exp(lams)

    # Build the model
    # Numerical functions given spline terms s(),
    # Categorical variables given step function terms f().
    gam_model = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+
                            f(7)+f(8)+f(9)+f(10)+f(11)+f(12)+f(13)+
                            f(14)).gridsearch(X_train, Y_train, n_splines=grid_splines, lam=lams)

    return gam_model



# Build MLR model
def build_mlr():

    mlr_model = LinearRegression()

    return mlr_model