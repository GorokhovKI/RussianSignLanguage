# realtime_demo.py
import cv2
import numpy as np
import torch
import mediapipe as mp


# Загрузка модели
checkpoint = torch.load('slovo_model.pth')
model = GestureLSTM(num_classes=1000)
model.load_state_dict(checkpoint['model_state'])
model.eval()

mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
cap = cv2.VideoCapture(0)
frame_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    frame_landmarks = []
    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            for lm in hand.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
    while len(frame_landmarks) < 126:
        frame_landmarks.extend([0.0, 0.0, 0.0])
    frame_buffer.append(frame_landmarks)
    # Когда накопилось N кадров, делать предсказание
    N = 30
    if len(frame_buffer) == N:
        inp = torch.tensor([frame_buffer], dtype=torch.float32)  # shape [1, N, 126]
        with torch.no_grad():
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).item()
        frame_buffer.pop(0)  # сдвигаем окно
        predicted_text = train_dataset.idx_to_class[pred]
        # Наложение текста на кадр (см. раздел 7)
    cv2.imshow('Sign Language Translation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
