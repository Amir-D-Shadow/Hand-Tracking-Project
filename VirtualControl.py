import cv2
import mediapipe as mp
import time
import numpy as np
from win32api import GetSystemMetrics
import HandTracking_min as htm
import math
from pynput.mouse import Button,Controller

#Detector Set up
scr_w,scr_h = GetSystemMetrics(0),GetSystemMetrics(1)

wCam,hCam = 1280,720
cap = cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.9,trackCon=0.55)

#volume set up
##from ctypes import cast, POINTER
##from comtypes import CLSCTX_ALL
##from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
##devices = AudioUtilities.GetSpeakers()
##interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
##volume = cast(interface, POINTER(IAudioEndpointVolume))
##volRange = volume.GetVolumeRange()
##minVol = volRange[0]
##maxVol = volRange[1]
#volume.SetMasterVolumeLevel(-20.0, None)

#Set up Mouse
mouse = Controller()
prevClickTime = 0

while cap.isOpened():

    cTime = time.time()
    
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)

    if len(lmList) != 0:

        #functional fingers
        x0,y0 = lmList[4][1],lmList[4][2]
        x1,y1 = lmList[8][1],lmList[8][2]
        x2,y2 = lmList[12][1],lmList[12][2]

        #reference origin
        ox,oy = lmList[0][1],lmList[0][2]

        #Set Mouse Position
        cv2.circle(img,(x1,y1),15,(0,69,255),2)
        hImg,WImg,_ = img.shape
        
        adapted_x = scr_w-int((scr_w/WImg)*x1)#np.interp(x1,[0,1280],[0,1920])
        adapted_y = int((scr_h/hImg)*y1*1.2)#np.interp(y1,[0,720],[0,1080])#
        mouse.position = (adapted_x,adapted_y)

        #print(adapted_x,adapted_y)
        
        #cv2.line(img,(x1,y1),(x2,y2),(0,69,255),2)
        
        dis1 = int(math.hypot(x0-ox,y0-oy))
        #print(dis1)

        #Activate Finger 4
        if dis1 > 100:

            cv2.circle(img,(x0,y0),15,(0,69,255),2)
            
            #length48 = int(math.hypot(x0-x1,y0-y1))
            #vol = np.interp(length48,[30,160],[minVol,maxVol])
            #volume.SetMasterVolumeLevel(vol, None)

        #Activate Finger 12
        dis2 = int(math.hypot(x2-ox,y2-oy))
        #print(dis2)

        if dis2 >100:
            
            cv2.circle(img,(x2,y2),15,(0,69,255),2)

            length = int(math.hypot(x1-x2,y1-y2))
            #print(length)
            if length < 30 :

                click_Timediff = cTime - prevClickTime

                if click_Timediff >= 0.5: 

                    mouse.click(Button.left,1)
                    prevClickTime = cTime

    
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    img = cv2.flip(img,1)
    cv2.putText(img,str(int(fps)),(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)


    cv2.imshow("IMG",img)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()
