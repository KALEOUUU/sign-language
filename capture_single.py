import cv2
import mediapipe as mp
import os
import time

DATA_DIR = "BISINDO_Dataset/Y"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = "Y"
dataset_size = 900

cap = cv2.VideoCapture(1)

if not os.path.exists(os.path.join(DATA_DIR, classes)):
    os.makedirs(os.path.join(DATA_DIR, classes))

print("Collecting data for class {}".format(classes))

while True:
    ret, frame = cap.read()
    cv2.putText(frame, "Press Q to start!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) == ord('q'):
        time.sleep(5)
        break

counter = 0 
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, '{}_{}.jpg'.format(classes, counter)), frame)

    counter += 1

