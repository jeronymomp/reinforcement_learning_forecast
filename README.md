# Reinforcement Learning Forecasting Toolkit

## Overview

This package suite is designed to simplify the implementation of Reinforcement Learning (RL) models for time series forecasting. It includes essential components for building agents, simulated training environments, and auxiliary tools for data processing and performance evaluation.

## Installation

To use this toolkit, clone the repository and import the necessary functions into your Python environment:

```python
from rl_forecasting_toolkit import * ```

## Packages

Functions Overview:

- series_rolling_window_list

Creates a list of rolling windows from a time series.
Example: Converts [1, 2, 3, 4, 5] into [[1, 2, 3], [2, 3, 4], [3, 4, 5]] with a window size of 3.

- training_series_for_pca

Extracts and reshapes a segment of time series data for PCA training.

- cosine_similarity_q_table

Computes cosine similarity between an embedding and each entry in a Q-table.

- q_learning_state_selection

Chooses a Q-table state based on similarity, blending states if necessary.

- q_learning_method_selection

Selects a forecasting method based on similarity and returns forecast values and errors.

- q_learning_table_update

Updates the Q-table with new values based on the expected Q-values and similarity measures.

- generate_Xt

Generates a list of values for simulation.

- mean_squared_dataframe

Generates a DataFrame with Mean Squared Error (MSE).