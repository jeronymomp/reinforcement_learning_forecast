# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Forecast Paper

This is the code for the Experiment number 8 of the pdf "Proposed SImulation Experiment"
"""

# =============================================================================
# 1 - Packages
# =============================================================================
# Base Functions
import os
import sys
import time
import datetime
import warnings
import itertools
import statistics
import numpy as np
import pandas as pd
import random as rand
from itertools import repeat
from numpy.linalg import norm
from datetime import datetime
from sklearn.decomposition import PCA  
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# 2 - Parameters
# =============================================================================
# Set seed
warnings.filterwarnings("ignore")
rand.seed(10)

# 2.1 Size of Training Sample for the RL
train_sample = 100 
T = 100

# 2.2 Initial date to simulate a time series
initial_date = '1973-01-01'

# 2.3 Parameters for Normal Distribution
mu1, mu2, sigma1, sigma2 = 0, 0, 1, 1

# 2.4 Number of Time Series to be generated
number_ts = 500

# 2.6 - Parameters for Reinforcement Learning
alpha = 0.9

# 2.7 - Number of Experiments
n_experiments = 1000

# 2.8 - Window size for rolling window
window_size = 3

# 2.9 - Observation to use for rolling window
first_rolling_obs = train_sample + 1 - window_size + 1

# 2.10 - Number of components for PCA
n_components = 3

# 2.11 - Start of the PCA embedding
start = 0

# 2.12 - Measures of similiarities
exploration_similarity = 0.7
state_similarity = 0.7
min_similarity = 0.7

# 2.13 Q-Table
q_table = []

# 2.15 - Number of observations
n_obs = 1000

# 2.14 - RL Dataframe
rl_data_frame = pd.DataFrame(np.nan, index=range(n_obs + T), columns=range(2))
rl_data_frame.columns = ['forecast','error']

# 2.15 - Path
package_path = '/data/keeling/a/jeronymo/reinforcement_learning_forecast/src'
result_path = '/data/keeling/a/jeronymo/reinforcement_learning_forecast/experiments/8'

# Result_list
result_df_mae_lst = []
result_df_mse_lst = []

# =============================================================================
# 3 - Built-in Functions
# =============================================================================
sys.path.append(package_path)
from rl_forecasting import series_rolling_window_list,training_series_for_pca,cosine_similarity_q_table,generate_Xt
from rl_forecasting import q_learning_state_selection,q_learning_method_selection,q_learning_table_update,mean_squared_dataframe,mean_absolute_dataframe

# =============================================================================
# 3 - Generate Many Monthly Simulated Series
# ============================================================================= 
sys.path.append(result_path)

# Start list
list_of_Xt = []
list_of_Xt_1 = []
list_of_Xt_2 = []
list_of_Yt = []

# Random Number of Series to be Combinated
n_series_ensamble = rand.randint(2,number_ts)

# Series that generates Yt
# Weak Alternatives
list_of_Xt_1 = generate_Xt(n_series_ensamble,mu1,sigma1,T,list_of_Xt_1,n_obs)
# X
X = np.column_stack(list_of_Xt_1)
# Y
Yt = np.mean(list_of_Xt_1,axis=0)
list_of_Yt.append(Yt) 
# Regression
reg_Y = LinearRegression().fit(X[0:T], Yt[0:T])
Y_base = reg_Y.predict(X[(T+1):])  
list_of_Xt_1.append(Y_base)

# Other Series
list_of_Xt_2 = generate_Xt(number_ts-n_series_ensamble,mu1,sigma1,T,list_of_Xt_2,n_obs)

# List of Dataframes of results
benchmark_forecast = []

# Included the Strong Model
for i in range(len(list_of_Xt_1)-1):
    reg_1 = LinearRegression().fit(list_of_Xt_1[i][0:T], list_of_Yt[0][0:T])
    y_pred_1 = reg_1.predict(list_of_Xt_1[i][(T+1):])
    benchmark_forecast.append(y_pred_1)

# Not Included the Strong Model
for i in range(len(list_of_Xt_2)):
    reg_2 = LinearRegression().fit(list_of_Xt_2[i][0:T], list_of_Yt[0][0:T])
    y_pred_2 = reg_2.predict(list_of_Xt_2[i][(T+1):])
    benchmark_forecast.append(y_pred_2)

# Append Last Regression Results
benchmark_forecast.append(Y_base)

# Mean Average of Forecasts
forec_mean_avg = np.mean(benchmark_forecast,axis=0)

# Combine the two lists of X
list_of_Xt = list_of_Xt_1 + list_of_Xt_2

# DataFrame of Results
data_frame = pd.DataFrame(list_of_Yt[0], columns=['sim_data'])

# Numpy array of zeros (no loop needed for initialization)
zeros = np.zeros(train_sample + 1)

# Add benchmark columns efficiently
forecast_array = np.concatenate([np.append(zeros, forecast)[:, None] for forecast in benchmark_forecast], axis=1)
forecast_df = pd.DataFrame(forecast_array, columns=[f'forc_{i}' for i in range(len(list_of_Xt))])
data_frame = pd.concat([data_frame, forecast_df], axis=1)

# Add mean forecast
data_frame['avg_forc'] = np.append(zeros, forec_mean_avg)

# Create Error DataFrame
data_frame_error = data_frame.filter(like='forc').subtract(data_frame['sim_data'], axis=0)
data_frame_error.columns = [f'{col}_error' for col in data_frame_error.columns]

# MSE DataFrame (assumindo que a função mean_squared_dataframe já está otimizada)
data_frame_mse = mean_squared_dataframe(data_frame_error, 0)

# Methods
list_of_methods = data_frame.columns
list_of_methods = list_of_methods[1:]

# =============================================================================
# 6 - State Definition
# =============================================================================  
# Base series for embedding
data = data_frame_mse[0:train_sample]
 
# PCA training
pca = PCA(n_components=n_components)
pca.fit(data)

# First Training Embedding
first_embed = data_frame_mse.iloc[train_sample,:]
first_embedding = pca.transform(np.array(first_embed).reshape(1, -1))

# Append to Q-Table
new_row = data_frame_mse.iloc[(train_sample),:]
new_tuple = (new_row, first_embedding)
q_table.append(new_tuple)

# =============================================================================
# 7 - Q-Learning
# =============================================================================  
for i in range((train_sample+1),len(data_frame_mse)):
	# Start Embedding
	embed = data_frame_mse.iloc[i,:]
	embedding = pca.transform(np.array(embed).reshape(1, -1))

	# Cosine Similarity
	sim_list,max_value,idx_max_value = cosine_similarity_q_table(embedding,q_table)

	# Q Table selection
	chosen_q_table = q_learning_state_selection(q_table,sim_list,max_value,idx_max_value,exploration_similarity)

	# Method selection
	method,new_line = q_learning_method_selection(chosen_q_table,max_value,exploration_similarity,'avg_forc','sim_data',i,data_frame)
	rl_data_frame.iloc[i,:] = new_line

	# Q-Table Update
	q_table = q_learning_table_update(alpha,q_table,chosen_q_table,idx_max_value,max_value,state_similarity,i,data_frame_mse)
 
# =============================================================================
# 7 - Forecasting errors
# =============================================================================
data_frame['reinforce'] = rl_data_frame['forecast']

# Calculate all MSE
mse_lst = []
data_frame_test = data_frame.iloc[(train_sample+100):,:]
for j in list_of_methods:
	mse = mean_squared_error(data_frame_test['sim_data'], data_frame_test[j])
	mse_row = (j, mse)
	mse_lst.append(mse_row)

# Add RL method
mse_rl = mean_squared_error(data_frame_test['sim_data'][train_sample+1:], data_frame_test['reinforce'][train_sample+1:])
mse_row = ('reinforce', mse_rl)
mse_lst.append(mse_row)

result_df_mse = pd.DataFrame(mse_lst, columns=['Method', 'Value'])

hour = datetime.now().strftime('%H-%M-%S-%f')
file_name = f"mse_{hour}.txt"
result_df_mse.to_csv(file_name, sep='\t', index=False)

# Calculate all MAE
mae_lst = []
data_frame_test = data_frame.iloc[(train_sample+100):,:]
for j in list_of_methods:
	mae = mean_absolute_error(data_frame_test['sim_data'], data_frame_test[j])
	mae_row = (j, mae)
	mae_lst.append(mae_row)

# Add RL method
mae_rl = mean_absolute_error(data_frame_test['sim_data'], data_frame_test['reinforce'])
mae_row = ('reinforce', mae_rl)
mae_lst.append(mae_row)

result_df_mae = pd.DataFrame(mae_lst, columns=['Method', 'Value'])
hour = datetime.now().strftime('%H-%M-%S-%f')
file_name = f"mae_{hour}.txt"
result_df_mae.to_csv(file_name, sep='\t', index=False)

# =============================================================================
