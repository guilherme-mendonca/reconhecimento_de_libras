import cv2
import mediapipe as mp
import csv
import os

# Pasta onde salvar
os.makedirs("dados", exist_ok=True)

# Inicializar captura e MediaPipe
video = cv2.VideoCapture(0)
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1, 
                  min_detection_confidence=0.7, 
                  min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

def extrair_pontos_norm(handLms):
    pontos = []
    coords = [(lm.x, lm.y) for lm in handLms.landmark]

    # Referência = pulso
    cx, cy = coords[0]
    rel_coords = [(x - cx, y - cy) for x, y in coords]

    # Normalizar pelo maior valor
    max_val = max(max(abs(x), abs(y)) for x, y in rel_coords)
    norm_coords = [(x / max_val, y / max_val) for x, y in rel_coords]

    for x, y in norm_coords:
        pontos.append(x)
        pontos.append(y)

    return pontos

# Nome da letra que estamos coletando
letra = input("Digite a letra que deseja coletar: ").upper()
arquivo_csv = f"dados/{letra}.csv"

with open(arquivo_csv, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        check, img = video.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = Hand.process(imgRGB)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, hand.HAND_CONNECTIONS)

                pontos = extrair_pontos_norm(handLms)

                if pontos:
                    writer.writerow(pontos)

        cv2.putText(img, f"Coletando: {letra}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Coletor Libras", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para parar
            break

video.release()
cv2.destroyAllWindows()
