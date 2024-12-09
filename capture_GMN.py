import cv2
import mediapipe as mp
import os
import time

DATA_DIR = "BISINDO_Dataset/GMN"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = ["G", "M", "N"]
dataset_size = [1, 5, 10]

cap = cv2.VideoCapture(1)

j = 0
for i in classes:
    if not os.path.exists(os.path.join(DATA_DIR, i)):
        os.makedirs(os.path.join(DATA_DIR, i))

    print("Collecting data for class {}".format(i))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Press Q to start!", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            time.sleep(5)
            break
    
    while j < len(dataset_size):
        counter = 0 
        while counter < dataset_size[j]:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, '{}_{}.jpg'.format(i, counter)), frame)

            counter += 1
        j += 1

        break

cap.release()
cv2.destroyAllWindows()