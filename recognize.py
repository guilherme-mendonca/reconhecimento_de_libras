# recognize.py <image_path>
import sys
import cv2
import mediapipe as mp
import joblib
import numpy as np

MODEL_PATH = "modelo_libras.pkl"  # coloque seu modelo aqui

def extrair_pontos_norm(handLms):
    pontos = []
    coords = [(lm.x, lm.y) for lm in handLms.landmark]
    cx, cy = coords[0]
    rel_coords = [(x - cx, y - cy) for x, y in coords]
    max_val = max(max(abs(x), abs(y)) for x, y in rel_coords) or 1.0
    norm_coords = [(x / max_val, y / max_val) for x, y in rel_coords]
    for x, y in norm_coords:
        pontos.append(x); pontos.append(y)
    return pontos

def main():
    if len(sys.argv) < 2:
        print("")
        return
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print("")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)

    if not res.multi_hand_landmarks:
        print("")  # nada detectado
        return

    # Carregar modelo
    clf = joblib.load(MODEL_PATH)

    for handLms in res.multi_hand_landmarks:
        pontos = extrair_pontos_norm(handLms)
        if pontos:
            probas = clf.predict_proba([pontos])[0]
            idx = int(np.argmax(probas))
            letra = clf.classes_[idx]
            conf = probas[idx]
            if conf > 0.5:
                print(letra)
                return
    print("")  # fallback

if __name__ == "__main__":
    main()
