import cv2
import os
import mediapipe as mp
import time
import string

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "data_vanue"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = list(string.ascii_uppercase)
dataset_size = 900

cap = cv2.VideoCapture(0)

# if not os.path.exists(os.path.join(DATA_DIR, classes)):
#     os.makedirs(os.path.join(DATA_DIR, classes))

n_landmark = [42,42,21,42,21,42,
              42,42,21,21,42,21,
              42,42,21,42,42,21,
              42,42,21,21,42,42,
              42,21]

for i, n in zip(classes, n_landmark):
    print("Collecting data for class {}".format(i))

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press Q to start!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3,
                    cv2.LINE_AA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        cv2.imshow('frame', frame)
        cv2.imshow('frame_rgb', frame_rgb)

        if cv2.waitKey(25) == ord('q'):
            time.sleep(10)
            break
    
    counter = 0 
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[lm].x
                    y = hand_landmarks.landmark[lm].y
                    z = hand_landmarks.landmark[lm].z
                    data_aux.append((x,y,z))

            if len(data_aux) == n:
                cv2.imshow('frame_rgb', frame_rgb)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(DATA_DIR, '{}_{}.jpg'.format(i, counter+1000)), frame)

                counter += 1