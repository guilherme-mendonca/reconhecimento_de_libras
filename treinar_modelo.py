import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

print("=== Treinando modelo Libras ===")

# Pasta com os CSVs coletados
DATASET_DIR = "dados"

X = []
y = []

# Ler todos os arquivos CSV de cada letra
for arquivo in os.listdir(DATASET_DIR):
    if arquivo.endswith(".csv"):
        letra = arquivo.replace(".csv", "")
        caminho = os.path.join(DATASET_DIR, arquivo)

        with open(caminho, "r") as f:
            reader = csv.reader(f)
            for linha in reader:
                if linha:  # evitar linhas vazias
                    X.append([float(val) for val in linha])
                    y.append(letra)

# Converter para numpy
X = np.array(X)
y = np.array(y)

print(f"Total de amostras: {len(X)}")

# Divisão treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar classificador (SVM)
clf = SVC(kernel="rbf", probability=True)  # agora ele calcula probabilidades
clf.fit(X_train, y_train)

# Acurácia
acc = clf.score(X_test, y_test)
print(f"Acurácia de teste: {acc*100:.2f}%")

# Salvar modelo
joblib.dump(clf, "modelo_libras.pkl")
print("Modelo salvo como 'modelo_libras.pkl'")
