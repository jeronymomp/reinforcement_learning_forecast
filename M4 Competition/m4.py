# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Forecast Paper

This is the code for the Experiment number 1 of the pdf "Proposed SImulation Experiment"
"""

# =============================================================================
# 1 - Packages
# =============================================================================
# Base Functions
import os
import time
import patoolib
import itertools
import statistics
import numpy as np
import pandas as pd
import random as rand
from numpy.linalg import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA  
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =============================================================================
# 2 - Parameters
# =============================================================================
# Set seed
rand.seed(10)

# 2.1 Size of Training Sample for the RL
train_sample = 10 
T = 100

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
n_obs = 48

# 2.14 - RL Dataframe
rl_data_frame = pd.DataFrame(np.nan, index=range(n_obs), columns=range(2))
rl_data_frame.columns = ['forecast','error']

# 2.15 - Path
package_path = 'C:/Git/Git Privado/reinforcement_learning_forecast/src'
m4_path = 'C:/Git/Git Privado/M4-methods'

# 2.16 - List of results
data_frame_lst = []
q_table_lst = []
result_df_lst_mae = []
result_df_lst_mse = []
rl_data_frame_lst = []

# =============================================================================
# 3 - Built-in Functions
# =============================================================================
os.chdir(package_path)
from rl_forecasting import series_rolling_window_list,training_series_for_pca,cosine_similarity_q_table,generate_Xt
from rl_forecasting import q_learning_state_selection,q_learning_method_selection,q_learning_table_update,mean_squared_dataframe,mean_absolute_dataframe

# Auxiliary Function
def process_competitor(i, hour, sim_data):
    submission_data = pd.read_csv('Point Forecasts/submission-' + i + '.csv', index_col=0)
    forecast = submission_data.loc['H' + str(hour)].values[:48]
    return forecast

# =============================================================================
# 3 - Import Data
# =============================================================================
os.chdir(m4_path)

# Info about the data to be forecast
data_info = pd.read_csv('Dataset/M4-info.csv')

# Info about the competitors
competitors_info = pd.read_excel('Point Forecasts/Submission Info.xlsx')

# Read Data
train_data = pd.read_csv('Dataset/Train/Hourly-train.csv',index_col=0)

test_data = pd.read_csv('Dataset/Test/Hourly-test.csv',index_col=0)

# Get indexes for loop
id_competitors = competitors_info['ID']
id_competitors = id_competitors.apply(lambda x: '{0:0>3}'.format(x))

# Extract .rar - commented for study cases
#for j in range(len(id_competitors)):
    #i = id_competitors[j]
    #patoolib.extract_archive('Point Forecasts\submission-'+ i + '.rar', outdir="Point Forecasts")

# Get a specific series data
for hour in range(1,414):
    sim_data = test_data.loc['H'+str(hour)].values
    
    # Create DataFrame with Values
    data_frame = pd.DataFrame()
    data_frame['sim_data'] = sim_data
    
    # Final DataFrame
    for j in range(len(id_competitors)):
        i = id_competitors[j]
        submission_data = pd.read_csv('Point Forecasts/submission-' + i + '.csv',index_col=0)
        forecast = submission_data.loc['H'+str(hour)].values
        data_frame['forc_' + i] = forecast[0:48]
       
    data_frame['mean_avg'] = data_frame.iloc[:,1:].mean(axis=1)  

    # Create Error Dataframe
    data_frame_error = pd.DataFrame()
    
    # Methods
    list_of_methods = data_frame.columns
    list_of_methods = list_of_methods[1:]
    
    for i in list_of_methods:
        data_frame_error[str(i) + '_error'] = data_frame['sim_data'] - data_frame[str(i)]
        
    # MSE Dataframe
    data_frame_mse = mean_absolute_dataframe(data_frame_error,(0)) 

# =============================================================================
# 4 - State Definition
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
# 5 - Q-Learning
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
        method,new_line = q_learning_method_selection(chosen_q_table,max_value,exploration_similarity,'mean_avg','sim_data',i,data_frame)
        rl_data_frame.iloc[i,:] = new_line
        
        # Q-Table Update
        q_table = q_learning_table_update(alpha,q_table,chosen_q_table,idx_max_value,max_value,state_similarity,i,data_frame_mse)
         
# =============================================================================
# 7 - Forecasting errors
# =============================================================================
    data_frame['reinforce'] = rl_data_frame['forecast']

    # Calculate all MSE
    mse_lst = []
    data_frame_test = data_frame.iloc[11:,:]
    for j in list_of_methods:
        mse = mean_squared_error(data_frame_test['sim_data'], data_frame_test[j])
        mse_row = (j, mse)
        mse_lst.append(mse_row)
        
    # Add RL method
    mse_rl = mean_squared_error(data_frame_test['sim_data'], data_frame_test['reinforce'])
    mse_row = ('reinforce', mse_rl)
    mse_lst.append(mse_row)
    
    result_df_mse = pd.DataFrame(mse_lst, columns=['Method', 'Value'])
    
    # Calculate all MAE
    mae_lst = []
    data_frame_test = data_frame.iloc[11:,:]
    for j in list_of_methods:
        mae = mean_absolute_error(data_frame_test['sim_data'], data_frame_test[j])
        mae_row = (j, mae)
        mae_lst.append(mae_row)
        
    # Add RL method
    mae_rl = mean_absolute_error(data_frame_test['sim_data'], data_frame_test['reinforce'])
    mae_row = ('reinforce', mae_rl)
    mae_lst.append(mae_row)
    
    result_df_mae = pd.DataFrame(mae_lst, columns=['Method', 'Value'])
    
# =============================================================================
# 8 - Saving Results
# =============================================================================
    data_frame_lst.append(data_frame)
    q_table_lst.append(q_table)
    result_df_lst_mae.append(result_df_mae)
    result_df_lst_mse.append(result_df_mse)
    rl_data_frame_lst.append(rl_data_frame)

# =============================================================================
# 9 - Checking results
# =============================================================================
result_df_lst_media = [df.set_index(df.columns[0])[df.columns[1]] for df in result_df_lst_mse]

rank_list = []

for rank in range(0,len(result_df_lst_media)):
    ranking = pd.DataFrame(result_df_lst_media[rank].rank(ascending=True))
    rank_list.append(ranking)
    
# Concatenar os DataFrames ao longo do eixo das colunas
df_concatenado = pd.concat(rank_list)

# Agrupar pelos índices e calcular a média
df_media = df_concatenado.groupby(df_concatenado.index).mean()

# =============================================================================
