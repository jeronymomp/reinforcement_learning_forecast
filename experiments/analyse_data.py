# =============================================================================
# Packages
# =============================================================================
import os
import pandas as pd

# =============================================================================
# Parameters
# =============================================================================
diretorio = "C:\\Git\\Git Privado\\reinforcement_learning_forecast\\experiments\\1"

# =============================================================================
# Functions
# =============================================================================
def read_error_frame(error_type,diretorio, delimitador='\t'):
    dataframes = []
    
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.startswith(error_type) and nome_arquivo.endswith(".txt"):
            caminho_completo = os.path.join(diretorio, nome_arquivo)
            df = pd.read_csv(caminho_completo, delimiter=delimitador)
            dataframes.append(df)
    
    return dataframes

# =============================================================================
# Read Content
# =============================================================================
lista_mae = read_error_frame("mae",diretorio)
lista_mse = read_error_frame("mse",diretorio)

# =============================================================================
# Get Mean Average
# =============================================================================
lista_mae = [df.set_index(df.columns[0]) for df in lista_mae]
lista_mse = [df.set_index(df.columns[0]) for df in lista_mse]

# Mean Average

# MAE
sum_mae = sum(lista_mae)
avg_mae = sum_mae / len(lista_mae)

# MSE
sum_mse = sum(lista_mse)
avg_mse = sum_mse / len(lista_mse)