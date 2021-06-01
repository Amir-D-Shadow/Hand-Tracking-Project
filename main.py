import HandTracking_min as htm
import cv2
import time
import mediapipe as mp

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

detector = htm.handDetector()

while cap.isOpened():

    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    
    if len(lmList) != 0:

        pass

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    img = cv2.flip(img,1)
    
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(100,255,0),3)

    cv2.imshow("Hand Tracking",img)

    if (cv2.waitKey(1) & 0xFF ) == ord('q'):

        break


cap.release()
cv2.destroyAllWindows()
