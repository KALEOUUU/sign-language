import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict21 = pickle.load(open("model21.p", 'rb'))
model21 = model_dict21['model']

model_dict42 = pickle.load(open("model42.p", 'rb'))
model42 = model_dict42['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

confidence_threshold = 0.35  # Threshold untuk confidence

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[1].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append((x, y, z))
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        if len(data_aux) == 42:
            proba = model42.predict_proba([np.asarray(data_aux).flatten()])
            prediction = model42.classes_[np.argmax(proba)]
            confidence = np.max(proba)
        
        elif len(data_aux) == 21:
            proba = model21.predict_proba([np.asarray(data_aux).flatten()])
            prediction = model21.classes_[np.argmax(proba)]
            confidence = np.max(proba)
        
        else:
            confidence = 0
            prediction = "No detection"
        
        if confidence >= confidence_threshold:
            cv2.putText(frame, f'{prediction} ({confidence*100:.2f}%)', (x1, y1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No detection", (x1, y1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
