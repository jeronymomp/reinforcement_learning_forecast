# Reinforcement Learning for Forecasting

The **Reinforcement Learning for Forecasting** package provides a comprehensive suite for implementing Reinforcement Learning (RL) models tailored specifically for forecasting tasks. Reinforcement Learning is a dynamic and powerful approach where agents learn to make optimal decisions by interacting with their environment. This methodology is particularly beneficial in forecasting, where it allows models to adaptively select, refine, and optimize their forecasting techniques based on observed performance and the evolving characteristics of time series data.

This package incorporates various methods explored in the research paper titled *"Reinforcement Learning: A New Paradigm for the Forecasting Combination Puzzle."* Through rigorous experimentation, we aim to demonstrate the effectiveness of RL in addressing complex forecasting challenges.

The package is organized into four main sections, each focusing on different aspects of our methodology and experimentation:

## M4

This section encompasses all the methods utilized during our experiments to evaluate the performance of our RL-based forecasting approach against established benchmarks from the M4 competition. The M4 competition is a widely recognized forecasting challenge that provides a rich dataset and a variety of forecasting methods. By comparing our RL methods with those employed in the M4 competition, we aim to assess the strengths and weaknesses of our approach in a competitive context. 

### Key Features:
- Detailed implementation of RL algorithms for time series forecasting.
- Benchmarks against M4 competition methods to validate performance.
- Comprehensive performance metrics for evaluating accuracy and reliability.

## SopF

In this section, we present all the methods implemented during our experiments to compare our RL-based forecasting strategy against the Survey of Professional Forecasters (SopF) methods. The SopF provides insights into how professional forecasters approach their predictions, and understanding these methods is crucial for assessing the practical applicability of our RL approach in real-world scenarios.

### Key Features:
- Implementation of various SopF forecasting methods as benchmarks.
- Performance evaluation to determine the effectiveness of RL against professional forecasting practices.
- Insights into how RL can enhance forecasting accuracy compared to traditional methods.

## Experiments

This section outlines all methods used during our experiments to test our RL approach against the proposed experimental setups. We detail our experimental design, including the datasets, evaluation criteria, and comparative methodologies employed in our analysis. This provides a clear framework for understanding the robustness of our findings and the potential implications of using RL in forecasting applications.

### Key Features:
- Comprehensive description of the experimental methodology.
- Evaluation of RL performance in diverse forecasting scenarios.
- Analysis of results to highlight the advantages and limitations of the proposed methods.

Through these sections, we aim to provide a thorough understanding of how Reinforcement Learning can be leveraged for effective forecasting, demonstrating its potential to revolutionize traditional forecasting practices.

## src - methods to apply RL in forecasting

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
