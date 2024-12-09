import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "BISINDO_Dataset/train/images"

data = []
labels = []

for img_path in os.listdir(DATA_DIR):
    data_aux = []

    img = cv2.imread(os.path.join(DATA_DIR, img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[1].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append((x,y,z))
        
        data.append(data_aux)
        labels.append(img_path[0])

f = open("data_ada_shabrinya.pickle",'wb')
pickle.dump({'data': data, 'labels':labels}, f)
f.close()
