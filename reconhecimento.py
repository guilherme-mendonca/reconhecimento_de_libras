import os
import time
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ===== Carrega modelo treinado =====
with open('modelo_libras.pkl', 'rb') as f:
    modelo = pickle.load(f)

# ===== Inicializa MediaPipe =====
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ===== Função para extrair pontos normalizados =====
def extrair_pontos_norm(handLms):
    pontos = []
    coords = [(lm.x, lm.y) for lm in handLms.landmark]
    cx, cy = coords[0]
    rel_coords = [(x - cx, y - cy) for x, y in coords]
    max_val = max(max(abs(x), abs(y)) for x, y in rel_coords)
    norm_coords = [(x / max_val, y / max_val) for x, y in rel_coords]
    for x, y in norm_coords:
        pontos.append(x)
        pontos.append(y)
    return pontos

# ===== Função para processar uma imagem e prever letra =====
def prever_letra(caminho_img):
    img = cv2.imread(caminho_img)
    if img is None:
        print(f"❌ Erro ao ler {caminho_img}")
        return None

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if not result.multi_hand_landmarks:
        print(f"⚠️ Nenhuma mão detectada em {os.path.basename(caminho_img)}")
        return None

    for handLms in result.multi_hand_landmarks:
        pontos = extrair_pontos_norm(handLms)
        X = np.array(pontos).reshape(1, -1)
        letra = modelo.predict(X)[0]
        return letra
    return None

# ===== Monitora pasta e forma palavras =====
pasta_uploads = "uploads"
arquivos_processados = set()
palavra_atual = ""

print("🧩 Monitorando novas imagens em:", os.path.abspath(pasta_uploads))
print("Pressione Ctrl+C para encerrar.\n")

try:
    while True:
        for arquivo in os.listdir(pasta_uploads):
            caminho = os.path.join(pasta_uploads, arquivo)
            if caminho not in arquivos_processados and arquivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                letra = prever_letra(caminho)
                if letra:
                    palavra_atual += letra
                    print(f"🔠 Letra detectada: {letra} | Palavra atual: {palavra_atual}")
                arquivos_processados.add(caminho)
        time.sleep(1)  # evita uso excessivo de CPU
except KeyboardInterrupt:
    print("\n🛑 Monitoramento encerrado.")
