# --------------------------------------------------------
# Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Variáveis globais

SHEET_URL = "16o3WCX6ULp16YCeNAdkAzZBG11yiEZjH"
SHEET_NAME = "01_aircraft_survey.xlsx"
POLY_DEGREE = 1

# --------------------------------------------------------
# Funções

def download_sheet():
    if SHEET_NAME in os.listdir():
        answer = input("Você quer baixar novamente a planilha? (S,N) \n")
        if answer in ['S', 's']:
            os.system("rm 01_aircraft_survey.xlsx")
            print("Baixando a planilha ...")
            os.system("gdown --id {}".format(SHEET_URL))
    else:
        print("Baixando a planilha ...")
        os.system("gdown --id {}".format(SHEET_URL))

def fit_label(z):
    name = "fit: $"
    for i in range(len(z)-1, -1, -1):
        if i > 1:
            if z[i] > 0.001:
                if name[-1] != "$" and z[i] > 0:
                    name += "+ "
                name += "{0:.3f}x^{1}".format(z[i], i)
        elif i == 1:
            if z[i] > 0.001:
                if name[-1] != "$" and z[i] > 0:
                    name += "+ "
                name += "{0:.3f}x".format(z[i])
        else:
            if z[i] > 0.001:
                if name[-1] != "$" and z[i] > 0:
                    name += "+ "
                name += "{0:.3f}".format(z[i])

    name += "$"
    return name

def create_plot(X, y, title, path_name, xlabel, ylabel):
    z = np.polyfit(X, y, POLY_DEGREE)
    p = np.poly1d(z)
    x_range = np.linspace(np.sort(X)[0], np.sort(X)[-1], 100)
    y_pred = p(np.sort(X))
    r_2 = r2_score(y, y_pred)

    plt.figure(figsize=(12,8))
    plt.yscale('log')
    plt.xscale('log')
    plt.scatter(X, y, label="Dados experimentais")
    name_fit = fit_label(z)
    plt.plot(x_range, p(x_range), color='red', label=name_fit)
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path_name)
    plt.show()

# --------------------------------------------------------
# 1. Download planilha




# Se não tiver gdown baixado basta abrir o terminal
# (ou o Anaconda terminal no Windows) e rodar:
# pip install gdown
print("1. Download planilha")
download_sheet()
print("Download Feito.")

# --------------------------------------------------------
# 2. Processamento dos dados

# FIle Read
print("2. Processamento dos dados")
df = pd.read_excel(SHEET_NAME)

# --------------------------------------------------------
# 3. Gerando os plots

print("3. Gerando os plots")
# Plot MTOW x Peso vazio
create_plot(X=df.iloc[:, 1], 
            y=df.iloc[:, 2], 
            title="Plot MTOW x Peso vazio", 
            path_name="MTOW_Peso_vazio.jpeg",
            xlabel=df.columns[1],
            ylabel=df.columns[2])


# Plot MTOW x Número de passageiros

create_plot(X=df.iloc[:, 1], 
            y=df.iloc[:, 4], 
            title="Plot MTOW x Número de passageiros", 
            path_name="MTOW_N_passageiros.jpeg",
            xlabel=df.columns[1],
            ylabel=df.columns[4])

# Plot Número de passageiros × Comprimento da fuselagem

create_plot(X=df.iloc[:, 35], 
            y=df.iloc[:, 4], 
            title="Plot Número de passageiros × Comprimento da fuselagem", 
            path_name="N_passageiros_Comprimento_da_fuselagem.jpeg",
            xlabel=df.columns[35],
            ylabel=df.columns[4])
