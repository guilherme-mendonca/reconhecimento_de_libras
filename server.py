from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os, time, traceback
import cv2, numpy as np
import mediapipe as mp
from joblib import load

# ---------------- App / Socket ----------------
app = Flask(__name__)
CORS(app)  # libera CORS p/ fetch do HTML no celular
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ---------------- Modelo / Estado -------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "modelo_libras.pkl")   # ajuste se o pkl estiver em outro lugar
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    modelo = load(MODELO_PATH)
    print(f"✅ Modelo carregado: {MODELO_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Erro ao carregar modelo_libras.pkl: {e}")

palavra_atual = ""
ultimo_tempo_mao = time.time()
tempo_limite_espaco = 4.0  # segundos sem mão => espaço

# ---------------- MediaPipe -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3
)

# ---------------- Utils ----------------------
def extrair_pontos_norm(handLms):
    coords = [(lm.x, lm.y) for lm in handLms.landmark]  # 21 landmarks
    cx, cy = coords[0]  # punho
    rel = [(x - cx, y - cy) for x, y in coords]
    max_val = max(max(abs(x), abs(y)) for x, y in rel) or 1.0
    feat = []
    for x, y in rel:
        feat += [x / max_val, y / max_val]
    return np.array(feat, dtype=np.float32).reshape(1, -1)

def corrigir_img(img):
    # rotação 180º (sua ESP32 estava invertida) + pequeno boost de brilho/contraste + resize
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
    h, w = img.shape[:2]
    if max(h, w) > 960:
        s = 960.0 / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)))
    return img

# ---------------- Endpoints -------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "ts": time.time()}), 200

@app.route("/reset", methods=["POST"])
def reset():
    global palavra_atual, ultimo_tempo_mao
    palavra_atual = ""
    ultimo_tempo_mao = time.time()
    socketio.emit("reset", {"palavra": palavra_atual})
    print("♻️ Reset acionado.")
    return jsonify({"status": "reset"}), 200

@app.route("/upload", methods=["POST"])
def upload():
    global palavra_atual, ultimo_tempo_mao
    try:
        raw = request.data
        if not raw:
            return jsonify({"error": "sem dados"}), 400

        npimg = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ JPEG inválido")
            return jsonify({"error": "jpeg invalido"}), 400

        # (opcional) salvar último frame p/ debug
        cv2.imwrite(os.path.join(UPLOAD_DIR, "ultimo_frame.jpg"), img)

        # corrigir e detectar
        img = corrigir_img(img)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(imgRGB)

        if not res.multi_hand_landmarks:
            dt = time.time() - ultimo_tempo_mao
            if dt > tempo_limite_espaco and not palavra_atual.endswith(" "):
                palavra_atual += " "
                ultimo_tempo_mao = time.time()
                socketio.emit("espaco", {"palavra": palavra_atual})
                print(f"🕒 Pausa → espaço | Palavra: '{palavra_atual}'")
            else:
                print("⚠️ Nenhuma mão detectada.")
            return jsonify({"status": "ok", "letra": None, "palavra": palavra_atual}), 200

        # extrair features e prever letra
        X = extrair_pontos_norm(res.multi_hand_landmarks[0])
        letra = str(modelo.predict(X)[0])
        palavra_atual += letra
        ultimo_tempo_mao = time.time()

        socketio.emit("nova_letra", {"letra": letra, "palavra": palavra_atual})
        print(f"🔠 Letra: {letra} | Palavra: '{palavra_atual}'")
        return jsonify({"status": "ok", "letra": letra, "palavra": palavra_atual}), 200

    except Exception:
        print("❌ Erro no /upload:\n", traceback.format_exc())
        return jsonify({"error": "erro interno"}), 500

# ---------------- Run ------------------------
if __name__ == "__main__":
    print("🚀 Servidor no ar (Socket.IO + eventlet). Endpoint: /upload")
    # Instale antes: pip install eventlet flask-socketio flask-cors
    socketio.run(app, host="0.0.0.0", port=5000)
