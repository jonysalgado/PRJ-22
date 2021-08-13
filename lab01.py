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

# --------------------------------------------------------
# Funções

def download_sheet():
    if "01_aircraft_survey.xlsx" in os.listdir():
        answer = input("Você quer baixar novamente a planilha? (S,N) \n")
        if answer in ['S', 's']:
            os.system("rm 01_aircraft_survey.xlsx")
            print("Baixando a planilha ...")
            os.system("gdown --id {}".format(SHEET_URL))
    else:
        print("Baixando a planilha ...")
        os.system("gdown --id {}".format(SHEET_URL))

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

# Plot MTOW x Peso vazio

X, y = df.iloc[:2, 1], df.iloc[:2, 2]
z = np.polyfit(X, y, 2)
p = np.poly1d(z)
y_pred = p(np.sort(X))
r_2 = r2_score(y, y_pred)

plt.figure(figsize=(12,8))
plt.scatter(X, y, label="fit")
plt.plot(np.sort(X), y_pred, color='red', label="Dados experimentais")
plt.grid(True)
plt.legend()
plt.title("Plot MTOW x Peso vazio | $R^2 = ${}".format(r_2))
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[2])
plt.show()


# Plot MTOW x Número de passageiros

X, y = df.iloc[:2, 1], df.iloc[:2, 4]
z = np.polyfit(X, y, 2)
p = np.poly1d(z)
y_pred = p(np.sort(X))
r_2 = r2_score(y, y_pred)

plt.figure(figsize=(12,8))
plt.scatter(X, y, label="fit")
plt.plot(np.sort(X), y_pred, color='red', label="Dados experimentais")
plt.grid(True)
plt.legend()
plt.title("Plot MTOW x Número de passageiros | $R^2 = ${}".format(r_2))
plt.xlabel(df.columns[1])
plt.ylabel(df.columns[4])
plt.show()

# Plot Número de passageiros × Comprimento da fuselagem

X, y = df.iloc[:2, 32], df.iloc[:2, 4]
z = np.polyfit(X, y, 2)
p = np.poly1d(z)
y_pred = p(np.sort(X))
r_2 = r2_score(y, y_pred)

plt.figure(figsize=(12,8))
plt.scatter(X, y, label="fit")
plt.plot(np.sort(X), y_pred, color='red', label="Dados experimentais")
plt.grid(True)
plt.legend()
plt.title("Plot Número de passageiros × Comprimento da fuselagem | $R^2 = ${}".format(r_2))
plt.xlabel(df.columns[32])
plt.ylabel(df.columns[4])
plt.show()