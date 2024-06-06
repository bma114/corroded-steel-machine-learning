from sanfis import SANFIS
from Functions import *

import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import catboost as cb # CatBoost package
# from pygam import LinearGAM, s, f # pyGAM package for linear regression GAMs




### ========================= ### DATA PREPARATION ### ========================= ###


# Load database
df = load_data()
cat_cols, num_cols, anfis_cols, out_col = col_type(df) # Separate features by type - lists


# CatBoost model - Keep categorical features as strings, only encode numericals.
X_ns = pd.DataFrame(feature_scaling(df[num_cols]))
X_cat = pd.concat([X_ns, df[cat_cols]], axis='columns')
X_cat = X_cat.to_numpy()


# Encode categorical variables for all other models
df[cat_cols] = encoder(df[cat_cols])
df[cat_cols] = df[cat_cols].astype(object) # Convert to object


# Feature Dataframe for all other models except Catboost and ANFIS
X = pd.concat([df[num_cols], df[cat_cols]], axis='columns')
X_s = feature_scaling(X) # Normalize all features
Y = df[out_col]
Y = Y.to_numpy()


# Feature Dataframe for ANFIS model
X_ANFIS = df[anfis_cols]
X_te_s = feature_scaling(X_ANFIS) # Normalize all features


# Convert Datasets to PyTorch Tensors for ANFIS model
X_te = X_te_s.astype(np.float32)
X_te = torch.from_numpy(X_te)
Y_te = Y.astype(np.float32)
Y_te = torch.from_numpy(Y_te)


### ========================= ### TRAIN & TEST MODELS ### ========================= ###
"""

Train, test and evaluate each model based on the full dataset and using a Repeated K-Fold cross-validation approach. 

Repeat the 10-fold cross-validation 10 times (100-fold total) reshuffling the dataset split each 10 folds.

The model performance metrics over all 100-folds and all models are exported as csv files at the end of the script.

Model optimization is excluded form this example to improve computational speed. However, code is provided 
at the end of the script for example implementation of optimization for several key model types. 
Optimization code is commented out by default. Uncomment the code to run the algorithms. 


"""

k_fold_init = 10

r2, mse, rmse, mae, mape = np.zeros([100,9]), np.zeros([100,9]), np.zeros([100,9]), np.zeros([100,9]), np.zeros([100,9])

# Empty lists for combining test sets and predictions
# Insert lists in best performing model to plot model regression, residuals, etc.
Y_test_all = []
y_best_all = [] 

time_start = time.time()

i = 0
fold_shuffle = np.random.randint(10,100,10)

for j in range(k_fold_init):
    
    kf = KFold(n_splits=10, random_state=fold_shuffle[j], shuffle=True) # Define the fold split and reshuffle each loop.
    
    for train_index, test_index in kf.split(X_s, Y):
        
        # All features are numerical and normalized
        X_train, X_test = X_s[train_index], X_s[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_inv = np.exp(Y_test) # Convert test set back into original scale -- from natural log

        # Training & Validation datasets for ANN
        X_70, X_val, Y_70, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

        # Datasets for ANFIS model - converted to PyTorch Tensors
        X_train_te, X_test_te = X_te[train_index], X_te[test_index]
        Y_train_te, Y_test_te = Y_te[train_index], Y_te[test_index]

        # Datasets for CatBoost - categorical features left as strings
        X_train_cat, X_test_cat = X_cat[train_index], X_cat[test_index]

        # Compile test data from all k-folds to plot best model
        Y_test_all.append(Y_inv)



        ### ========================= ### NEURAL NETWORKS ### ========================= ###
        # ### =============================== ### ANN ### =============================== ###

        # Call ANN model
        ann_model = ann_architecture()

        # Train and evaluate the model
        history = ann_model.fit(X_train, Y_train, batch_size=25, epochs=150, verbose=0, validation_data=(X_val, Y_val))
        ann_model.evaluate(X_test, Y_test)

        # Generate predictions
        y_pred_ann = ann_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_ann_inv = np.exp(y_pred_ann)

        # Record error metrics from each fold
        r2[i,0] = r_squared(Y_inv, y_ann_inv)
        mse[i,0] = mean_squared_err(Y_inv, y_ann_inv)
        rmse[i,0] = root_mean_squared_err(Y_inv, y_ann_inv)
        mae[i,0] = mean_abs_err(Y_inv, y_ann_inv)
        mape[i,0] = mean_abs_perc_err(Y_inv, y_ann_inv)



        ### ========================= ### ANFIS ### ========================= ###
        
        # Call ANFIS functions
        MEMBFUNCS, anfis_params = fis_params() # Set hyperparameters and membership functions through Functions.py
        fis = SANFIS(membfuncs=MEMBFUNCS, n_input=8, scale=anfis_params['scaler']) # All other ANFIS model code for backend.
        
        loss_functions = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(fis.parameters(), lr=anfis_params['lr'])

        # fit model
        history = fis.fit([X_train_te, Y_train_te], [X_test_te, Y_test_te],
                          optimizer=optimizer, loss_function=loss_functions, batch_size=anfis_params['batch_size'],
                          epochs=anfis_params['n_epochs'])

        # Predict data
        y_pred_anfis = fis.predict(X_test_te)

        # Convert prediction back to original magnitude
        y_pred_anfis = y_pred_anfis.numpy()
        y_anfis_inv = np.exp(y_pred_anfis)   

        # Record error metrics from each fold
        r2[i,1] = r_squared(Y_inv, y_anfis_inv)
        mse[i,1] = mean_squared_err(Y_inv, y_anfis_inv)
        rmse[i,1] = root_mean_squared_err(Y_inv, y_anfis_inv)
        mae[i,1] = mean_abs_err(Y_inv, y_anfis_inv)
        mape[i,1] = mean_abs_perc_err(Y_inv, y_anfis_inv)



        ### ========================= ### TREE MODELS ### ========================= ###
        ### ============================ ### GBRT ### ============================= ### 
        
        # Call GBRT model
        gbrt_model = build_gbrt()    

        # Train the optimised model
        gbrt_model.fit(X_train, Y_train.ravel())

        # Predict the response 
        y_pred_gbrt = gbrt_model.predict(X_test)


        # Convert prediction back to original magnitude
        y_gbrt_inv = np.exp(y_pred_gbrt).reshape(-1,1)

        # After running 100 iterations, GBRT is the best performing model.
        # Record all predictions into an array for plotting
        y_best_all.append(y_gbrt_inv)

        # Record error metrics from each fold
        r2[i,2] = r_squared(Y_inv, y_gbrt_inv)
        mse[i,2] = mean_squared_err(Y_inv, y_gbrt_inv)
        rmse[i,2] = root_mean_squared_err(Y_inv, y_gbrt_inv)
        mae[i,2] = mean_abs_err(Y_inv, y_gbrt_inv)
        mape[i,2] = mean_abs_perc_err(Y_inv, y_gbrt_inv)



        ### ========================= ### CBR ### ========================= ###
      
        # Call the CBR model
        cb_model = build_cb()

        # Format training and testing sets
        cb_train = cb.Pool(X_train_cat, Y_train, cat_features=[7,8,9,10,11,12,13,14])
        cb_test = cb.Pool(X_test_cat, Y_test, cat_features=[7,8,9,10,11,12,13,14])
               
        # Fit the model
        cb_model.fit(cb_train)

        # Generate predictions
        y_pred_cb = cb_model.predict(X_test_cat)

        # Convert prediction back to original magnitude
        y_cb_inv = np.exp(y_pred_cb).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,3] = r_squared(Y_inv, y_cb_inv)
        mse[i,3] = mean_squared_err(Y_inv, y_cb_inv)
        rmse[i,3] = root_mean_squared_err(Y_inv, y_cb_inv)
        mae[i,3] = mean_abs_err(Y_inv, y_cb_inv)
        mape[i,3] = mean_abs_perc_err(Y_inv, y_cb_inv)



        ### ========================= ### RF ### ========================= ###

        # Call RF model
        rf_model = build_rf()

        # Train the model
        rf_model.fit(X_train, Y_train.ravel())

        # Predict response
        y_pred_rf = rf_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_rf_inv = np.exp(y_pred_rf).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,4] = r_squared(Y_inv, y_rf_inv)
        mse[i,4] = mean_squared_err(Y_inv, y_rf_inv)
        rmse[i,4] = root_mean_squared_err(Y_inv, y_rf_inv)
        mae[i,4] = mean_abs_err(Y_inv, y_rf_inv)
        mape[i,4] = mean_abs_perc_err(Y_inv, y_rf_inv)



        ### ========================= ### KERNEL-BASED MACHINES ### ========================= ###
        ### ================================== ### SVR ### ================================== ###

        # Call SVR model
        svr_model = build_svr()

        # Train the model
        svr_model.fit(X_train, Y_train.ravel())

        # Predict the response
        y_pred_svr = svr_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_svr_inv = np.exp(y_pred_svr).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,5] = r_squared(Y_inv, y_svr_inv)
        mse[i,5] = mean_squared_err(Y_inv, y_svr_inv)
        rmse[i,5] = root_mean_squared_err(Y_inv, y_svr_inv)
        mae[i,5] = mean_abs_err(Y_inv, y_svr_inv)
        mape[i,5] = mean_abs_perc_err(Y_inv, y_svr_inv)



        ### ========================= ### GPR ### ========================= ###

        # Call GPR model
        gpr_model = build_gpr()

        # Fit the model
        gpr_model.fit(X_train, Y_train)
        gpr_params = gpr_model.kernel_.get_params() # Outputs the tune kernel function hyperparameters

        # Predict the response
        y_pred_gpr, std = gpr_model.predict(X_test, return_std=True)

        # Convert prediction back to original magnitude
        y_gpr_inv = np.exp(y_pred_gpr).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,6] = r_squared(Y_inv, y_gpr_inv)
        mse[i,6] = mean_squared_err(Y_inv, y_gpr_inv)
        rmse[i,6] = root_mean_squared_err(Y_inv, y_gpr_inv)
        mae[i,6]= mean_abs_err(Y_inv, y_gpr_inv)
        mape[i,6] = mean_abs_perc_err(Y_inv, y_gpr_inv)



        ### ========================= ### LINEAR MODELING ### ========================= ###
        ### =============================== ### GAM ### =============================== ###

        gam_model = build_gam(X_train, Y_train)

        # Train the model
        gam_model.fit(X_train, Y_train)

        # Predict the response
        y_pred_gam = gam_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_gam_inv = np.exp(y_pred_gam).reshape(-1,1)

        # Record error metrics from each fold
        r2[i,7] = r_squared(Y_inv, y_gam_inv)
        mse[i,7] = mean_squared_err(Y_inv, y_gam_inv)
        rmse[i,7] = root_mean_squared_err(Y_inv, y_gam_inv)
        mae[i,7] = mean_abs_err(Y_inv, y_gam_inv)
        mape[i,7] = mean_abs_perc_err(Y_inv, y_gam_inv)



        ### ========================= ### MLR ### ========================= ###

        mlr_model = build_mlr()

        # Fit the model
        mlr_model.fit(X_train, Y_train)

        # Predict the response
        y_pred_mlr = mlr_model.predict(X_test)

        # Convert prediction back to original magnitude
        y_mlr_inv = np.exp(y_pred_mlr)

        # Record error metrics from each fold
        r2[i,8] = r_squared(Y_inv, y_mlr_inv)
        mse[i,8] = mean_squared_err(Y_inv, y_mlr_inv)
        rmse[i,8] = root_mean_squared_err(Y_inv, y_mlr_inv)
        mae[i,8] = mean_abs_err(Y_inv, y_mlr_inv)
        mape[i,8] = mean_abs_perc_err(Y_inv, y_mlr_inv)

    
        i += 1 
    j += 1
    

time_end = time.time()
print("Elapsed time: {} minutes and {:.0f} seconds".format
      (int((time_end - time_start) // 60), (time_end - time_start) % 60)) 




## ========================= ### MODEL ERROR AND PERFORMANCE ### ========================= ###


df_r2 = pd.DataFrame(r2).rename(columns={0:'ANN', 1:'ANFIS', 2:'GBRT', 
                                         3:'CBR', 4:'RF', 5:'SVR', 
                                         6:'GPR', 7:'GAM', 8:'MLR'})
df_mse = pd.DataFrame(mse).rename(columns={0:'ANN', 1:'ANFIS', 2:'GBRT', 
                                         3:'CBR', 4:'RF', 5:'SVR', 
                                         6:'GPR', 7:'GAM', 8:'MLR'})


df_rmse = pd.DataFrame(rmse).rename(columns={0:'ANN', 1:'ANFIS', 2:'GBRT', 
                                         3:'CBR', 4:'RF', 5:'SVR', 
                                         6:'GPR', 7:'GAM', 8:'MLR'})


df_mae = pd.DataFrame(mae).rename(columns={0:'ANN', 1:'ANFIS', 2:'GBRT', 
                                         3:'CBR', 4:'RF', 5:'SVR', 
                                         6:'GPR', 7:'GAM', 8:'MLR'})


df_mape = pd.DataFrame(mape).rename(columns={0:'ANN', 1:'ANFIS', 2:'GBRT', 
                                         3:'CBR', 4:'RF', 5:'SVR', 
                                         6:'GPR', 7:'GAM', 8:'MLR'})


# Export model performance results to excel files
df_r2.to_excel('100-fold R2 - All Models.xlsx')
df_mse.to_excel('100-fold MSE - All Models.xlsx')
df_rmse.to_excel('100-fold RMSE - All Models.xlsx')
df_mae.to_excel('100-fold MAE - All Models.xlsx')
df_mape.to_excel('100-fold MAPE - All Models.xlsx')

print(df_r2['GBRT'].mean()) # Change 'MLR' to any model to see mean metric
print(df_r2.head(20)) # First 20 rows R^2 dataframe





### ========================= ### HYPERPARAMETER OPTIMIZATION EXAMPLES ### ========================= ###
### ================================== ### ANN OPTIMIZATION ### ==================================== ###

# def ann_architecture(activation): # Create ANN model architecture
#     ann_model = Sequential()
#     ann_model.add(Dense(units = 128, input_dim=15, kernel_initializer='normal', activation='relu'))
#     ann_model.add(Dense(units = 64, kernel_initializer='normal', activation=activation)) 
#     ann_model.add(Dense(1))
#     ann_model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])    
#     return ann_model

# ann_model = KerasRegressor(build_fn = ann_architecture)

# # GridSearchCV Optimization
# params_ann = {'batch_size': [25,50,75], 
#               'epochs': [75,100,125,150,175], 
#               'activation': ['relu', 'sigmoid']}

# # Fit the model to the hyperparameter grid-search
# ann_opt = GridSearchCV(estimator = ann_model, param_grid=params_ann, 
#                                scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)
# ann_opt.fit(X_train, Y_train, verbose=0)

# print(" Results from ANN Grid Search " )
# print("\n The best score across ALL searched params:\n", ann_opt.best_score_)
# print("\n The best parameters across ALL searched params:\n", ann_opt.best_params_)



### ========================= ### GBRT OPTIMIZATION ### ========================= ###

# def gbrt_bench(): # Create benchmark GBRT model
#     gbrt_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=2, max_leaf_nodes=5,
#                                       min_samples_leaf=1, min_samples_split=2, random_state=0, loss='squared_error')
#     return gbrt_model

# gbrt_bench = gbrt_bench()

# # Train the benchmark model
# gbrt_bench.fit(X_train, Y_train.ravel())

# # Apply grid optimisation for other hyperparamters
# params_gbrt = {'n_estimators': [100,200,300,400,500], 
#                'learning_rate': [0.05,0.1,0.15,0.2,0.25], 
#                'max_depth': [2,4,6,8], 
#                'min_samples_split': [2,4,6,8]}

# # Fit the model to the hyperparameter grid-search
# gbrt_opt = GridSearchCV(estimator=gbrt_bench, param_grid=params_gbrt, cv=5, n_jobs=-1)
# gbrt_opt.fit(X_train, Y_train.ravel())

# print(" Results from GBRT Grid Search " )
# print("\n The best estimator across ALL searched params:\n", gbrt_opt.best_estimator_)
# print("\n The best score across ALL searched params:\n", gbrt_opt.best_score_)
# print("\n The best parameters across ALL searched params:\n", gbrt_opt.best_params_)



### ========================= ### CatBoost OPTIMIZATION ### ========================= ###

# cb_train = cb.Pool(X_train_cat, Y_train, cat_features=[7,8,9,10,11,12,13,14])
# cb_test = cb.Pool(X_test_cat, Y_test, cat_features=[7,8,9,10,11,12,13,14])

# # Build model
# cb_model = cb.CatBoostRegressor(iterations=1000, loss_function='MAE')

# # Define grid-search parameters
# cb_grid = {'iterations': [300,400,500], 'learning_rate': [0.05,0.1,0.15,0.2],
#              'depth':[2,4,6,8], 'l2_leaf_reg': [0.25,0.5,1]}

# # Fit the model to the hyperparameter grid-search
# cb_model.grid_search(cb_grid, cb_train, verbose=False)

# print("Elapsed time: %.2f seconds" % (time_end - time_start))
# print("Count of trees in model = {}".format(cb_model.tree_count_))
# print("\n The best parameters across ALL searched params:\n", cb_model.get_params())



### ========================= ### RF OPTIMIZATION ### ========================= ###

# def rf_bench():
#     rf_model = RandomForestRegressor(n_estimators=500, max_depth=5, max_features='auto', min_samples_leaf=1, 
#                                      min_samples_split=2, random_state=25, n_jobs=-1, criterion='squared_error')
#     return rf_model

# rf_model = rf_bench()

# # Define grid search parameters
# params_rf = {'n_estimators': [100,200,300,400,500], 
#              'max_features': [1.0, 'sqrt', 'log2'], 
#              'max_depth': [3,4,5,6,7]}

# # Fit the model to the hyperparameter grid-search
# rf_opt = GridSearchCV(estimator=rf_model, param_grid=params_rf, cv=5, n_jobs=-1)
# rf_opt.fit(X_train, Y_train)

# print(" Results from RF Grid Search " )
# print("\n The best estimator across ALL searched params:\n", rf_opt.best_estimator_)
# print("\n The best score across ALL searched params:\n", rf_opt.best_score_)
# print("\n The best parameters across ALL searched params:\n", rf_opt.best_params_)



### ========================= ### GPR OPTIMIZATION ### ========================= ###
# The kernel hyperparameters are automatically tuned in GaussianProcessRegressor, however, the model hyperparamters still need tuning.
# Note: return_std cannot be used with GridSearchCV.


# kernel = ConstantKernel(1.0, (1e-1, 1e3)) * RBF(10, (1e-3, 1e3))

# gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

# # Try gridsearchcv
# params_gpr = {'n_restarts_optimizer': [5,10,15,20], 'alpha': [0.01,0.1,1]}

# gpr_model = GridSearchCV(gpr_model, param_grid=params_gpr, cv=10, verbose=2, n_jobs=-1)

# # Fit the model
# gpr_model.fit(X_train, Y_train)

# # gpr_params = gpr_model.kernel_.get_params() # Outputs the tuned kernel function hyperparameters

# print("Elapsed time: %.2f seconds" % (time_end - time_start))
# print("\n The best parameters across ALL searched params:\n", gpr_model.best_params_)
# print("Optimized Kernel Hyperparameters: " )
# # print(gpr_params)
