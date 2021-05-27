import cv2
import mediapipe as mp
import time
import  numpy as np

#set up
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

prev_frame_time = 0
curr_frame_time = 0

#Open camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        flag,frame = cap.read()

        #BGR to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Detection
        results = hands.process(image)

        #RGB to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        #Rendering Result
        if results.multi_hand_landmarks:

            for num,hand in enumerate(results.multi_hand_landmarks):

                mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS)


        #flip horizontally
        image = cv2.flip(image,1)
        
        #Calculate FPS
        font = cv2.FONT_HERSHEY_SIMPLEX

        curr_frame_time = time.time()

        fps = 1/(curr_frame_time-prev_frame_time)
        prev_frame_time = curr_frame_time

        fps = int(fps)
        fps = str(fps)

        cv2.putText(image,fps,(10,100),font,3,(100,255,0),3,cv2.LINE_AA)

        cv2.imshow("Hand Tracking",image)

        if (cv2.waitKey(10) & 0xFF ) == ord('q'):

            break

cap.release()
cv2.destroyAllWindows()
