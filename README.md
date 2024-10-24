# Reinforcement Learning for Forecasting

This package suite simplifies the implementation of Reinforcement Learning (RL) models specifically designed for forecasting tasks. Reinforcement Learning is a powerful paradigm that enables agents to learn optimal strategies through interactions with their environment. In the context of forecasting, this approach allows models to adaptively select and refine their forecasting methods based on the observed performance and changing dynamics of time series data.

## Overview

The **Reinforcement Learning for Forecasting** package offers tools and methodologies to implement RL techniques in time series forecasting. It includes components for building agents, creating simulated training environments, and provides auxiliary tools for data processing and performance evaluation. The main goal of this package is to empower data scientists and researchers to leverage RL in improving forecasting accuracy and robustness.

Key features of this package include:

- **Rolling Window Creation**: Generate rolling windows from time series data for more granular analysis and modeling.
- **PCA Preparation**: Extract and reshape segments of time series data to facilitate PCA training, enhancing dimensionality reduction and feature extraction.
- **Cosine Similarity Computation**: Measure similarity between embeddings and entries in a Q-table, allowing for informed decision-making in method selection.
- **Q-learning Methods**: Utilize Q-learning strategies to dynamically select and update forecasting methods based on similarity measures and performance metrics.
- **Time Series Generation**: Create synthetic ARMA series for simulation and testing purposes.
- **Performance Metrics**: Calculate Mean Squared Error (MSE) and provide comprehensive error evaluations to assess model performance.

## Installation

To install this package, you can use pip:

```bash
pip install your-package-name
```

Replace `your-package-name` with the actual name of your package.

## Usage

Here's a brief overview of the main functions included in the package:

### `series_rolling_window_list(serie, window_size)`

Creates a list of rolling windows from a time series.

**Parameters:**
- `serie`: The input time series as a `pd.Series`.
- `window_size`: The size of the rolling window.

**Returns:**
- A list of rolling windows.

### `training_series_for_pca(data, start, first_rolling_obs, window_size)`

Extracts and reshapes a segment of time series data for PCA training.

**Parameters:**
- `data`: The input time series data as a `np.array` or `pd.Series`.
- `start`: The starting index for the segment.
- `first_rolling_obs`: Value of the last observation for the training data.
- `window_size`: The size of the rolling window.

**Returns:**
- A 2D array suitable for PCA training.

### `cosine_similarity_q_table(embedding, q_table)`

Computes cosine similarity between an embedding and each entry in a Q-table.

**Parameters:**
- `embedding`: The reference embedding vector as a `np.array`.
- `q_table`: A list of tuples containing actions and embedding vectors.

**Returns:**
- A tuple with a list of similarities, the maximum similarity, and the index of that entry.

### `q_learning_state_selection(...)`

Selects a state from the Q-table based on cosine similarity.

### `q_learning_method_selection(...)`

Selects a forecasting method based on the similarity value and returns the forecast and error.

### `q_learning_table_update(...)`

Updates the Q-table based on the expected Q-values and the similarity measure.

### `generate_Xt(number_ts, mu, sigma1, T, list_of_Xt, n_obs)`

Generates random time series data and appends it to a list.

### `mean_squared_dataframe(data_frame_error, start_obs)`

Calculates cumulative MSE for a dataframe of errors starting from a specific observation.

## Examples

```python
import your_package_name as rl_forecast

# Create a rolling window
rolling_windows = rl_forecast.series_rolling_window_list(your_series, window_size=3)

# Generate time series data
time_series_list = []
rl_forecast.generate_Xt(number_ts=5, mu=0, sigma1=1, T=100, list_of_Xt=time_series_list, n_obs=100)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
