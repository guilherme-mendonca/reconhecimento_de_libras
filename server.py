from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import time
from flask_socketio import SocketIO

# CONFIGURAÇÃO BÁSICA
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Carregar modelo treinado
modelo = load("modelo_libras.pkl")

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
palavra_atual = ""
ultimo_tempo_mao = time.time()
tempo_limite_espaco = 4.0  # segundos sem detectar mão = espaço

# Funções auxiliares
def extrair_pontos_norm(handLms):
    coords = [(lm.x, lm.y) for lm in handLms.landmark]
    cx, cy = coords[0]
    rel_coords = [(x - cx, y - cy) for x, y in coords]
    max_val = max(max(abs(x), abs(y)) for x, y in rel_coords)
    norm_coords = [(x / max_val, y / max_val) for x, y in rel_coords]
    pontos = []
    for x, y in norm_coords:
        pontos.append(x)
        pontos.append(y)
    return pontos

def reconhecer_letra(caminho_img):
    global palavra_atual, ultimo_tempo_mao

    img = cv2.imread(caminho_img)
    if img is None:
        print(f"❌ Erro ao ler {caminho_img}")
        return None

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if not result.multi_hand_landmarks:
        tempo_desde_ultima_mao = time.time() - ultimo_tempo_mao
        if tempo_desde_ultima_mao > tempo_limite_espaco:
            palavra_atual += " "
            print(f"🕒 Pausa detectada → espaço adicionado à palavra. ({tempo_desde_ultima_mao:.1f}s)")
            ultimo_tempo_mao = time.time()
        print("⚠️ Nenhuma mão detectada.")
        return None

    for handLms in result.multi_hand_landmarks:
        pontos = extrair_pontos_norm(handLms)
        X = np.array(pontos).reshape(1, -1)
        letra = modelo.predict(X)[0]
        palavra_atual += letra
        ultimo_tempo_mao = time.time()
        print(f"🔠 Letra detectada: {letra} | Palavra atual: {palavra_atual}")
        socketio.emit("nova_letra", {"letra": letra, "palavra": palavra_atual})
        return letra


# Rotas Flask
@app.route("/")
def home():
    return "🚀 Servidor Flask ativo no Render e pronto!"

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "file" not in request.files:
            return jsonify({"status": "erro", "mensagem": "Nenhum arquivo recebido"}), 400

        file = request.files["file"]
        filename = "captura.jpg"
        caminho = os.path.join(UPLOAD_FOLDER, filename)
        file.save(caminho)
        print(f"📸 Imagem salva: {filename}")

        letra = reconhecer_letra(caminho)
        if letra:
            return jsonify({"status": "ok", "letra": letra, "palavra": palavra_atual}), 200
        else:
            return jsonify({"status": "ok", "mensagem": "Nenhuma mão detectada"}), 200

    except Exception as e:
        print("❌ Erro ao processar:", e)
        return jsonify({"status": "erro", "mensagem": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
