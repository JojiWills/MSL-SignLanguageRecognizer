import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

def image_processed(hand_img):
    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    try:
        if output.multi_hand_landmarks:
            for landmarks in output.multi_hand_landmarks:
                for landmark in landmarks.landmark:
                    # You can access landmark coordinates here
                    x, y, z = landmark.x, landmark.y, landmark.z
                    # Convert normalized coordinates to pixel coordinates
                    h, w, _ = hand_img.shape
                    x_pixel, y_pixel = int(x * w), int(y * h)
                    # Draw a circle to annotate the landmark
                    cv2.circle(hand_img, (x_pixel, y_pixel), 5, (0, 255, 0), -1)

        # Close MediaPipe Hands
        hands.close()

        return hand_img
    except:
        return hand_img

# The rest of your code remains unchanged



    try:
        data = output.multi_hand_landmarks[0]
        #print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])

import pickle
# load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,921600))
    print(y_pred)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (50, 100)
    
    # fontScale
    fontScale = 3
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 5
    
    # Using cv2.putText() method
    frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()