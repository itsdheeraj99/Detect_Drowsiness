#Importing necessary libraries and functions

import dlib #for face recognition & facial landmarks
import cv2 #for face recognition
import numpy as np #for N-dimensional array
import pygame #for playing alarm sound
import imutils
from imutils import face_utils
import threading
from threading import Thread
from scipy.spatial import distance #for calculating the...
#...Eye_Aspect_Ratio(EAR)

#Function for calling the Alarm file and activating it
def sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


#Function for the calculation of eye aspect ratio
def eye_aspect_ratio(eye):
#compute the Euclidean distance between the vertical landmarks of eye
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

#compute the Euclidean distance between the horizontal landmarks of eye
    C = distance.euclidean(eye[0], eye[3])

#compute the Eye Eye_Aspect_Ratio
    ear = (A + B)/(2 * C)
    return ear


#declaring the EAR threshold value for eye blink and the number of
#consecutive frames for which the eye was closed
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 30 

#initializing the frame counter and the boolean for alarm control
counter = 0
ALARM_ON = True

#Initializing the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path of shape predictor file)

#grab the facial landmark indexes for the right and left eyes
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]

#Initializing the camera.
video_cap = cv2.VideoCapture(0)

while True:
#Initializing the videostream and, then converting it to
#gray colour channel
    ret,video = video_cap.read()
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

#detecting the faces in the video live stream
    faces = detector(gray)

    for face in faces:
    #Extracting the landmarks of the face region and then converting those
    #to N-dimensional array using Numpy
        landmark = predictor(gray,face)
        landmark = face_utils.shape_to_np(landmark)

    #Extracting the co-ordinates of eyes and calculating the EAR of each eye
        lefteye = landmark[lstart:lend]
        righteye = landmark[rstart:rend]
        leftEAR = eye_aspect_ratio(lefteye)
        rightEAR = eye_aspect_ratio(righteye)

    #Taking the average of both EAR
        ear = (leftEAR + rightEAR)/2

        #leftEyeHull = cv2.convexHull(lefteye)
        #rightEyeHull = cv2.convexHull(righteye)
        #cv2.drawContours(video, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(video, [rightEyeHull], -1, (0, 255, 0), 1)

        #Comparing the real-time EAR value with the threshold value
        if ear < EYE_AR_THRESH:
            counter = counter + 1

            if counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(video, "DROWSINESS ALERT!!!", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        else:
            counter = 0


    #displaying the EAR value
        cv2.putText(video, "EAR: {:.2f}".format(ear),(700,30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    #displaying the no. of frames the eyes is closed 
        cv2.putText(video, "COUNTER: {}".format(counter), (10,300),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    #displaying the live video stream and to exit the live videostream
    cv2.imshow("Drowsiness_detection", video)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cv2.destroyAllWindows()





