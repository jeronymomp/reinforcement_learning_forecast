# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Forecast Paper

This is the code for the Experiment number 1 of the pdf "Proposed SImulation Experiment"
"""

# =============================================================================
# 1 - Packages
# =============================================================================
# Base Functions
import time
import itertools
import statistics
import numpy as np
import pandas as pd
import random as rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Built-in functions
from rl_forecasting import series_rolling_window_list,training_series_for_pca,cosine_similarity_q_table,generate_Xt
from rl_forecasting import q_learning_state_selection,q_learning_method_selection,q_learning_table_update,mean_squared_dataframe

# =============================================================================
# 2 - Parameters
# =============================================================================
# Set seed
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
window_size = 10

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
n_obs = 1500

# 2.14 - RL Dataframe
rl_data_frame = pd.DataFrame(np.nan, index=range(n_obs + T), columns=range(2))
rl_data_frame.columns = ['forecast','error']

# =============================================================================
# Starting Loop
# =============================================================================
for _ in itertools.repeat(None, n_experiments):

# =============================================================================
# 3 - Generate Many Monthly Simulated Series
# ============================================================================= 
    # Start list
    list_of_Xt = []
    list_of_Yt = []
    
    # Generate series
    # Base Series
    Xt = np.array(np.random.normal(mu1, sigma1, (n_obs + T))).reshape((-1, 1))
    list_of_Xt.append(Xt)
    Yt =  Xt + np.array(np.random.normal(mu2, sigma2, (n_obs + T))).reshape((-1, 1))
    list_of_Yt.append(Yt) 
    
    # Weak Alternatives
    list_of_Xt = generate_Xt(number_ts,mu1,sigma1,T,list_of_Xt,n_obs)
    
# =============================================================================
# 4 - Time Series Forecasting Base Models
# =============================================================================
    # List of Dataframes of results
    benchmark_forecast = []

    # Included the Strong Model
    for i in range(len(list_of_Xt)):
        reg = LinearRegression().fit(list_of_Xt[i][0:T], list_of_Yt[0][0:T])
        y_pred = reg.predict(list_of_Xt[i][(T+1):])
        benchmark_forecast.append(y_pred)
        
    # Mean Average of Forecasts
    forec_mean_avg = np.mean(benchmark_forecast,axis=0)
    
# =============================================================================
# 5 - DataFrame of Results to RL
# =============================================================================
    data_frame = pd.DataFrame(list_of_Yt[0].tolist())
    data_frame.columns = ['sim_data']
    
    # Numpy array of zeros
    zeros = np.zeros(train_sample+1)
    
    # Adding Benchamark columns
    for i in range(len(list_of_Xt)):
        data_frame['forc_'+ str(i)] = np.append(zeros,benchmark_forecast[i])
    
    data_frame['avg_forc'] = np.append(zeros,forec_mean_avg)
    
    # Create Error Dataframe
    data_frame_error = pd.DataFrame()
    
    for i in range(len(list_of_Xt)):
        data_frame_error['forc_'+ str(i) + '_error'] = data_frame['sim_data'] - data_frame['forc_' + str(i)]
        
    data_frame_error['forc_avg_error'] = data_frame['sim_data'] - data_frame['avg_forc']
 
    # MSE Dataframe
    data_frame_mse = mean_squared_dataframe(data_frame_error,(train_sample+1))
    
# =============================================================================
# 6 - State Definition
# =============================================================================      
    # Base series for embedding
    serie = data_frame['sim_data']
    
    # Create a list of rolling windows series
    data = series_rolling_window_list(serie,window_size)
    
    # Training data for PCA
    first_rolling_obs = train_sample + 1 - window_size
    train_data = training_series_for_pca(data,start,first_rolling_obs,window_size)
    
    # PCA training
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    
    # First Training Embedding
    first_embed = data[first_rolling_obs]
    first_embedding = pca.transform(np.array(first_embed).reshape(1, -1))
    
    # Append to Q-Table
    new_row = data_frame_mse.iloc[(train_sample),:]
    new_tuple = (new_row, first_embedding)
    q_table.append(new_tuple)
    
# =============================================================================
# 7 - Q-Learning
# =============================================================================      
    for i in range((first_rolling_obs+1),len(data)):

        # Start Embedding
        embed = data[i]
        embedding = pca.transform(np.array(embed).reshape(1, -1))
        
        # Cosine Similarity
        sim_list,max_value,idx_max_value = cosine_similarity_q_table(embedding,q_table)
        
        # Q Table selection
        chosen_q_table = q_learning_state_selection(q_table,sim_list,max_value,idx_max_value,exploration_similarity)
        
        # Method selection
        method,new_line = q_learning_method_selection(chosen_q_table,max_value,exploration_similarity,'avg_forc','sim_data',(i+window_size-1),data_frame)
        rl_data_frame.iloc[(i+window_size-1),:] = new_line
        
        # Q-Table Update
        q_table = q_learning_table_update(alpha,q_table,chosen_q_table,idx_max_value,max_value,state_similarity,(i+window_size-2-train_sample),data_frame_mse)
        
# =============================================================================
# 7 - Forecasting errors
# =============================================================================
    # MSE for RL
    rl_data_frame_error = pd.DataFrame(rl_data_frame.iloc[:,1])
    rl_mse_cum = mean_squared_dataframe(rl_data_frame_error,(train_sample+1))
    rl_mse = rl_mse_cum.tail(1)
    rl_mse.index = ['reinforce']

    # MSE for other methods
    result_df = pd.DataFrame(data_frame_mse.iloc[-1,:])
    result_df.columns = ['error']
    result_df = pd.concat([result_df, rl_mse], axis=0)    

# =============================================================================
