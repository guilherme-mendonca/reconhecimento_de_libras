import cv2
import mediapipe as mp
import joblib
import numpy as np

print("=== Reconhecimento em tempo real Libras ===")

# Carregar modelo treinado
clf = joblib.load("modelo_libras.pkl")
classes = clf.classes_

# Inicializar captura de vídeo e MediaPipe
video = cv2.VideoCapture(0)
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1, 
                  min_detection_confidence=0.7, 
                  min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def extrair_pontos_norm(handLms):
    """Transforma landmarks em coordenadas normalizadas"""
    pontos = []
    coords = [(lm.x, lm.y) for lm in handLms.landmark]

    # Ponto de referência = pulso (id 0)
    cx, cy = coords[0]

    # Transformar em coordenadas relativas
    rel_coords = [(x - cx, y - cy) for x, y in coords]

    # Normalizar pelo maior valor (escala)
    max_val = max(max(abs(x), abs(y)) for x, y in rel_coords)
    norm_coords = [(x / max_val, y / max_val) for x, y in rel_coords]

    # Colocar em uma lista 1D
    for x, y in norm_coords:
        pontos.append(x)
        pontos.append(y)

    return pontos

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Hand.process(imgRGB)
    h, w, _ = img.shape
    pontos = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, hand.HAND_CONNECTIONS)

            # Extrair pontos normalizados
            pontos = extrair_pontos_norm(handLms)

        if pontos:
            probas = clf.predict_proba([pontos])[0]
            idx = np.argmax(probas)
            letra_pred = classes[idx]
            conf = probas[idx]

            if conf > 0.7:
                texto = f"Libras: {letra_pred} ({conf*100:.1f}%)"
            else:
                texto = "Libras: NULO"

            cv2.putText(img, texto, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow("Reconhecimento Libras", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

video.release()
cv2.destroyAllWindows()
