# -*- coding: utf-8 -*-
"""
This package suite simplifies the implementation of Reinforcement Learning models for forecasting.
It includes components for building agents, simulated training environments, 
and auxiliary tools for data processing and performance evaluation.

Methods:

series_rolling_window_list: Creates a list of rolling windows from a time series.

training_series_for_pca: Extracts and reshapes a segment of time series data for PCA training.

cosine_similarity_q_table: Computes cosine similarity between an embedding and each entry in a Q-table.

q_learning_state_selection: Chooses a Q-table state based on similarity, blending states if necessary.

q_learning_method_selection: Selects a forecasting method based on similarity and returns forecast values and errors.

q_learning_table_update: Updates the Q-table with new values based on the expected Q-values and similarity measures.

generate_Xt: generate a list of values to simulation.

mean_squared_dataframe: genarate dataframe with MSE.

"""

# =============================================================================
# Packages
# =============================================================================
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

# =============================================================================
# Function 1
# =============================================================================
def series_rolling_window_list(serie, window_size):
    """
    Generates a list of rolling windows from a time series.

    Args:
        serie (pd.Series): The input time series.
        window_size (int): The size of the rolling window.

    Returns:
        list: A list of lists, each containing a rolling window of the specified size.

    Example:
        Given a time series [1, 2, 3, 4, 5] with a window size of 3, 
        this function returns [[1, 2, 3], [2, 3, 4], [3, 4, 5]].
    """
    # Rolling Window
    list_rolling_series = [window.tolist() for window in serie.rolling(window=window_size) if len(window) == window_size]

    return list_rolling_series

# =============================================================================
# Function 2
# =============================================================================
def training_series_for_pca(data,start,first_rolling_obs,window_size):
    """
    Prepares training data for PCA by extracting a segment of the time series.

    Args:
        data (np.array or pd.Series): The input time series data.
        start (int): The starting index for the segment to be used in training.
        first_rolling_obs: value of the list on the rolling list that has the last obs as the last value for the training data.
    Returns:
        np.array: A 2D array where rows represent observations in the time series, 
        and columns represent the values within a rolling window.

    Example:
        If the input data is an array of size 1000 and start is 0, 
        this function will extract and reshape the data starting from index 0 
        into a format suitable for PCA.
    """
    # Extracting the segment of the time series
    train_data = np.array([
        data[start:first_rolling_obs]
    ])
    
    # Removing any unnecessary dimensions
    train_data = np.squeeze(train_data)
    
    # Reshaping the data to fit the rolling window format
    train_data = train_data.reshape(first_rolling_obs, window_size)
    
    return train_data

# =============================================================================
# Function 3
# =============================================================================
def cosine_similarity_q_table(embedding, q_table):
    """
    Calculates the cosine similarity between a given embedding and each entry in a Q-table.

    Args:
        embedding (np.array): The reference embedding vector.
        q_table (list): A list of tuples where each tuple contains an action and an embedding vector.

    Returns:
        tuple: 
            - sim_list (list): A list of cosine similarities between the given embedding and each embedding in the Q-table.
            - max_value (float): The maximum cosine similarity found.
            - idx_max_value (int): The index of the Q-table entry with the maximum cosine similarity.

    Example:
        Given an embedding and a Q-table, this function computes the cosine similarity between the embedding 
        and each entry in the Q-table, returning a list of similarities, the highest similarity, 
        and the index of the entry with that highest similarity.
    """
    # List of similarities
    sim_list = []
    for i in range(0,len(q_table)):
        embedding2 = q_table[i][1]
        cosine_similarity = np.dot(embedding, embedding2.T)/(norm(embedding)*norm(embedding2.T))
        cosine_similarity = [item[0] for item in cosine_similarity]
        sim_list.append(cosine_similarity)
    # Max value of similarity
    max_value = max(sim_list)
    # Index of max value of similarity
    idx_max_value = sim_list.index(max_value)
    
    return sim_list,max_value,idx_max_value

# =============================================================================
# Function 4
# =============================================================================
def q_learning_state_selection(q_table, sim_list, max_value, idx_max_value, min_similarity):
    """
    Selects a state from the Q-table based on cosine similarity, with an option to blend states if similarity is too low.

    Args:
        q_table (list): A list of tuples where each tuple contains an action and an embedding vector.
        sim_list (list): A list of cosine similarities between the current embedding and those in the Q-table.
        max_value (float): The maximum cosine similarity found.
        idx_max_value (int): The index of the Q-table entry with the maximum cosine similarity.
        min_similarity (float): The threshold for minimum acceptable similarity.

    Returns:
        list: A list containing a single tuple with the selected or blended state.

    Example:
        If the maximum similarity is below the threshold, the function blends the states in the Q-table 
        based on their similarities. Otherwise, it selects the state with the highest similarity.
    """
    chosen_q_table = []
    
    if max_value[0] < min_similarity:
        # List of weights
        total_sum = sum([item[0] for item in sim_list])
        normalized_lst = [[item[0] / total_sum] for item in sim_list]
        normalized_lst = [item for sublist in normalized_lst for item in sublist]
        # Separate q_table
        series = [sublist[0] for sublist in q_table]
        embeddings = [sublist[1] for sublist in q_table]
        # Mean Values
        mean_series = sum(w * v for w, v in zip(normalized_lst, series))
        mean_embedding = sum(w * v for w, v in zip(normalized_lst, embeddings))
        # Append to new list
        new_tuple = (mean_series, mean_embedding)
        chosen_q_table.append(new_tuple)
        
    else:
        # Just take the max similarity values
        series = q_table[idx_max_value][0]
        embedding = q_table[idx_max_value][1]
        new_tuple = (series, embedding)
        chosen_q_table.append(new_tuple)
        
    return chosen_q_table

# =============================================================================
# Function 5
# =============================================================================
def q_learning_method_selection(chosen_q_table, max_value, exploration_similarity, ensamble_method, original_series_name, observation_to_forecast, data_frame):
    """
    Selects a forecasting method based on the similarity value and returns the forecast and error.

    Args:
        chosen_q_table (list): The list of chosen Q-table states.
        max_value (float): The maximum cosine similarity value.
        exploration_similarity (float): The threshold for exploration similarity.
        ensamble_method (str): The method to use for forecasting if similarity is below the threshold.
        original_series_name (str): The name of the original series for error calculation.
        observation_to_forecast (int): The index of the observation to forecast.
        data_frame (pd.DataFrame): The DataFrame containing the series and methods for forecasting.

    Returns:
        tuple: 
            - chosen_method (str): The selected forecasting method.
            - new_value (list): A list containing the forecasted value and the forecast error.

    Example:
        If the maximum similarity is below the exploration similarity, the function uses the ensemble method 
        to forecast and calculate the error. Otherwise, it selects the method with the minimum error in the Q-table.
    """
    # If value has not significant similarity, use a mean
    if max_value[0] < exploration_similarity:
        chosen_method = ensamble_method
        forecast_value = data_frame[chosen_method][observation_to_forecast]
        forecast_error = data_frame[original_series_name][observation_to_forecast] - forecast_value
        new_value = [forecast_value,forecast_error]
    # Case with significant similarity => just replace old result
    else:
        chosen_method = chosen_q_table[0][0].idxmin()
        i1 = chosen_method.index('_')
        i2 = chosen_method.index('_', i1 + 1)
        chosen_method = chosen_method[:i2]
        forecast_value = data_frame[chosen_method][observation_to_forecast]
        forecast_error = data_frame[original_series_name][observation_to_forecast] - forecast_value
        new_value = [forecast_value,forecast_error]
        
    return chosen_method,new_value

# =============================================================================
# Function 6
# =============================================================================
def q_learning_table_update(alpha, q_table, chosen_q_table, idx_max_value, max_value, state_similarity, observation_to_forecast, data_frame_mse):
    """
    Updates the Q-table based on the expected Q-values and the similarity measure.

    Args:
        alpha (float): The learning rate for updating the Q-values.
        q_table (list): The current Q-table, a list of tuples where each tuple contains an action and an embedding vector.
        chosen_q_table (list): The list of chosen Q-table states.
        idx_max_value (int): The index of the Q-table entry with the maximum similarity.
        max_value (float): The maximum cosine similarity value.
        state_similarity (float): The threshold for state similarity to determine the update strategy.
        observation_to_forecast (int): The index of the observation used for the update.
        data_frame_mse (pd.DataFrame): The DataFrame containing the MSE values for the observations.

    Returns:
        list: The updated Q-table.

    Example:
        If the maximum similarity is below the state similarity threshold, the function appends a new entry to the Q-table.
        Otherwise, it updates the entry with the maximum similarity.
    """
    # If the similarity is not enough - mean of old values
    if max_value[0] < state_similarity:
        new_results = data_frame_mse.iloc[(observation_to_forecast),:]       
        expected_q = chosen_q_table[0][0] - alpha*(chosen_q_table[0][0] - new_results)
        new_values = (expected_q,chosen_q_table[0][1])
        q_table.append(new_values)  
    # Otherwise => just replace old value
    else: 
        new_results = data_frame_mse.iloc[(observation_to_forecast),:] 
        expected_q = chosen_q_table[0][0] - alpha*(chosen_q_table[0][0] - new_results)
        new_values = (expected_q,chosen_q_table[0][1])
        q_table[idx_max_value] = new_values
        
    return q_table

# =============================================================================
# Function 7
# =============================================================================
def generate_Xt(number_ts,mu,sigma1,T,list_of_Xt,n_obs):
    """
    Generate time series data and append it to a list.

    This function generates random time series data with a normal distribution
    characterized by the mean (`mu`) and standard deviation (`sigma1`). It creates
    `number_ts` time series, each with `T` data points, and appends them to the
    provided list `list_of_Xt`.

    Parameters:
    - number_ts (int): The number of time series to generate.
    - mu (float): The mean of the normal distribution for data generation.
    - sigma1 (float): The standard deviation of the normal distribution for data generation.
    - T (int): The number of data points in each time series.
    - list_of_Xt (list): A list to which the generated time series will be appended.

    Returns:
    None

    Example usage:
    ```python
    # Create an empty list to store time series data
    time_series_list = []

    # Generate 5 time series with mean 0, standard deviation 1, and 100 data points each
    generate_Xt(5, 0, 1, 100, time_series_list)

    # The time_series_list will now contain 5 time series data arrays.
    ```

    Note:
    - The generated time series data is in the form of NumPy arrays.
    - Each time series is a 2D array with shape (T, 1).
    """
    for i in range(0,number_ts):
        Xt = np.array(np.random.normal(mu, sigma1, (n_obs +T))).reshape((-1, 1))
        list_of_Xt.append(Xt)
    return list_of_Xt

# =============================================================================
# Function 8
# =============================================================================
def mean_squared_dataframe(data_frame_error,start_obs):
    """
    Calculates the cumulative mean squared error (MSE) for a dataframe of errors starting from a specific observation.

    The function takes a dataframe of errors and, starting from a specified observation, computes the absolute values of the errors,
    then calculates the cumulative sum of these values. The result is then divided by the index + 1 to obtain the cumulative MAE.

    Args:
        data_frame_error (pd.DataFrame): DataFrame containing the prediction errors.
        start_obs (int): Index of the starting observation from which the calculation will be performed.

    Returns:
        pd.DataFrame: DataFrame with the cumulative mean absolute error (MAE) starting from the specified observation.
    """
    df_test = data_frame_error.iloc[(start_obs):,:]
    df_squared = df_test ** 2
    df_cumulutive = df_squared.cumsum().reset_index(drop=True)
    data_frame_mse = df_cumulutive.div(df_cumulutive.index + 1, axis=0)
    
    return data_frame_mse

# =============================================================================
# Function 9
# =============================================================================
def mean_absolute_dataframe(data_frame_error, start_obs):
    """
    Calculates the cumulative mean absolute error (MAE) for a dataframe of errors starting from a specific observation.

    The function takes a dataframe of errors and, starting from a specified observation, computes the absolute values of the errors,
    then calculates the cumulative sum of these values. The result is then divided by the index + 1 to obtain the cumulative MAE.

    Args:
        data_frame_error (pd.DataFrame): DataFrame containing the prediction errors.
        start_obs (int): Index of the starting observation from which the calculation will be performed.

    Returns:
        pd.DataFrame: DataFrame with the cumulative mean absolute error (MAE) starting from the specified observation.
    """
    df_test = data_frame_error.iloc[start_obs:, :]
    df_absolute = df_test.abs()
    df_cumulative = df_absolute.cumsum().reset_index(drop=True)
    data_frame_mae = df_cumulative.div(df_cumulative.index + 1, axis=0)
    
    return data_frame_mae

# =============================================================================
# Function 10 - Optional
# =============================================================================
def q_learning_state_selection_2(q_table, sim_list, max_value, idx_max_value, min_similarity):
    """
    Selects a state from the Q-table based on cosine similarity, with an option to blend states if similarity is too low.

    Args:
        q_table (list): A list of tuples where each tuple contains an action and an embedding vector.
        sim_list (list): A list of cosine similarities between the current embedding and those in the Q-table.
        max_value (float): The maximum cosine similarity found.
        idx_max_value (int): The index of the Q-table entry with the maximum cosine similarity.
        min_similarity (float): The threshold for minimum acceptable similarity.

    Returns:
        list: A list containing a single tuple with the selected or blended state.
    """
    if max_value[0] < min_similarity:
        # Blending states when max similarity is below the threshold
        total_sum = sum(sim_list)
        if total_sum == 0:
            return [(0, 0)]  # Handle case where all similarities are zero
        normalized_lst = [sim / total_sum for sim in sim_list]

        series = [action for action, _ in q_table]
        embeddings = [embedding for _, embedding in q_table]

        mean_series = sum(w * v for w, v in zip(normalized_lst, series))
        mean_embedding = sum(w * v for w, v in zip(normalized_lst, embeddings))
        
        return [(mean_series, mean_embedding)]
    
    # Select the state with the highest similarity
    return [(q_table[idx_max_value][0], q_table[idx_max_value][1])]


# =============================================================================
# =============================================================================

