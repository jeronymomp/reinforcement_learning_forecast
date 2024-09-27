# -*- coding: utf-8 -*-
"""
Deep Reinforcement Learning Forecast Paper

This is the code for the SUrvey of Professional Forecasters Experiment."
"""

# =============================================================================
# 1 - Packages
# =============================================================================
# Base Functions
import os
import re
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

# 2.15 - Path
package_path = 'C:/Git/Git Privado/reinforcement_learning_forecast/src'
sopf_path = 'C:/Git/Git Privado/reinforcement_learning_general_framework/data/SofPF Series'

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
def make_dataframe(individual_forec,actual_series):
    """
    This function performs evrythong that is already done in other codes to perform
    all steps to reinforcement learning forecasting.
    
    The difference is that, encapsulating as function will allow us to do grid search
    over all parameters.
    """
    # 1 - Individual Forecasts
    # Import Data
    individual_forecast = pd.read_excel(individual_forec + '.xlsx')
    individual_forecast_grouped = pd.DataFrame(individual_forecast.dropna().groupby(by=["INDUSTRY","YEAR","QUARTER"],as_index=False)[variable + horizon].mean())
    # Individual Industry Forecasts
    # Mean
    if_df = individual_forecast_grouped[individual_forecast_grouped['INDUSTRY']==1]
    if_df_1 = individual_forecast_grouped[individual_forecast_grouped['INDUSTRY']==2]
    if_df_2 = individual_forecast_grouped[individual_forecast_grouped['INDUSTRY']==3]
    # Merge 
    # Mean
    if_df = pd.merge(if_df,if_df_1, on=['YEAR','QUARTER'])
    if_df = pd.merge(if_df,if_df_2, on=['YEAR','QUARTER'])
    # Colnames
    # Mean
    if_df = if_df[if_df.columns.drop(list(if_df.filter(regex='INDUSTRY')))]
    if_df.columns = ['YEAR','QUARTER',variable + horizon +'_1',variable + horizon +'_2',variable + horizon +'_3']
    del if_df_1, if_df_2
    # 2 - Real Data
    data_frame = pd.read_excel(actual_series + '.xlsx')
    data_frame = data_frame.iloc[:,np.r_[0, -1]]
    # Adjust Date Column
    date = data_frame['DATE'].str.split(':', n=1, expand=True)
    date.columns = ['YEAR', 'QUARTER']
    date['QUARTER'] = date['QUARTER'].str[1:]
    date = date.apply(pd.to_numeric)
    # Get Values
    values = pd.DataFrame(data_frame.iloc[:,1])
    values.columns = ['sim_data']
    # Final Data
    data_frame = pd.concat([date,values],axis=1)
    # Join with forecasts
    data_frame = pd.merge(data_frame,if_df, on=['YEAR','QUARTER']) 
    return data_frame

# =============================================================================
# 3 - Import Data
# =============================================================================
os.chdir(sopf_path)

files = os.listdir()
new_file = []

# Substitute ".xlsx" for nothing
for string in files:
    string2 = re.sub(".xlsx","",string)
    new_file.append(string2)
    
newlist = [x for x in new_file if not x.startswith('Individual_')]
newlist = [x for x in newlist if not x.startswith('List of Contents')]

# Search for variables
for sopf_data in newlist:
    q_table = []
    actual_series = variable = sopf_data
    individual_forec = 'Individual_' + actual_series
    horizon = '2'
    
    data_frame = make_dataframe(individual_forec,actual_series)
    data_frame = data_frame.iloc[:,2:]
    data_frame.columns = ['sim_data', '_1', '_2', '_3']

    # Calcular a média das previsões
    data_frame['mean_avg'] = data_frame.iloc[:, 1:].mean(axis=1)

    # Criar DataFrame de erros usando vetorização
    data_frame_error = data_frame.iloc[:, 1:].sub(data_frame['sim_data'], axis=0)
    data_frame_error.columns = [f'{col}_error' for col in data_frame_error.columns]

    # Calcular o MSE de forma vetorizada
    data_frame_mse = mean_absolute_dataframe(data_frame_error, (0))
    
    # Methods
    list_of_methods = data_frame.columns
    list_of_methods = list_of_methods[1:]
    
    # RL Dataframe
    n_obs = len(data_frame_mse)
    rl_data_frame = pd.DataFrame(np.nan, index=range(n_obs), columns=range(2))
    rl_data_frame.columns = ['forecast','error']

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
result_df_lst_mse = [df.set_index(df.columns[0])[df.columns[1]] for df in result_df_lst_mse]
result_df_lst_mae = [df.set_index(df.columns[0])[df.columns[1]] for df in result_df_lst_mae]

# Df mse raking
rank_list = []

for rank in range(0,len(result_df_lst_mse)):
    ranking = pd.DataFrame(result_df_lst_mse[rank].rank(ascending=True))
    rank_list.append(ranking)
    
# Concatenar os DataFrames ao longo do eixo das colunas
df_concatenado = pd.concat(rank_list)

# Agrupar pelos índices e calcular a média
df_mse_rank = df_concatenado.groupby(df_concatenado.index).mean()


# Df mse raking
rank_list = []

for rank in range(0,len(result_df_lst_mae)):
    ranking = pd.DataFrame(result_df_lst_mae[rank].rank(ascending=True))
    rank_list.append(ranking)
    
# Concatenar os DataFrames ao longo do eixo das colunas
df_concatenado = pd.concat(rank_list)

# Agrupar pelos índices e calcular a média
df_mae_rank = df_concatenado.groupby(df_concatenado.index).mean()

# To csv
df_mae_rank.to_excel('mae.xlsx')
df_mse_rank.to_excel('mse.xlsx')

# =============================================================================
