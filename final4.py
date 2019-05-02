#import open cv for image processing
import cv2  #it is the vision module 
import numpy as np  
import os
import sqlite3
import time

# Detect object in video stream using Haarcascade Frontal Face
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer\\trainingData.yml')


def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM people WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile





#here give font and color of the text
font=cv2.FONT_HERSHEY_COMPLEX
#start capturing video
cam=cv2.VideoCapture(0)
# Start looping
while(True):
    # Capture video frame
    ret,im=cam.read()
    # Convert frame to grayscale
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Detect frames of different sizes, list of faces rectangles
    faces=faceDetect.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    #faces = faceDetect.detectMultiScale(gray, 1.2,5)
    # Loops for each faces
    for(x,y,w,h) in faces:
        
        # Crop the image frame into rectangle
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        profile=getProfile(id)#call the method
        
        if(profile!=None):
            cv2.putText(im,"Name:"+str(profile[1]),(x,y+h+30),font,1,(0,250,0))
            cv2.putText(im,"Age:"+str(profile[2]),(x,y+h+60),font,1,(200,255,0))
            cv2.putText(im,"Gender:"+str(profile[3]),(x,y+h+90),font,1,(100,255,0))
            cv2.putText(im,"Criminal Records:"+str(profile[4]),(x,y+h+120),font,1,(0,255,255))
        #initialize cuerrent date
            currentDate= time.strftime("%d_%m_%y")
        #open file and store it into attn.txt folder
        with open("attn.txt","a+") as f:
            f.write("Name: "+str(profile[1])+" attendance on "+currentDate+"\n")

        
    # Display the video frame with the bounded rectangle    
    cv2.imshow("Face",im)
    


    
    # If 'q' is pressed, close program
    if(cv2.waitKey(1)==ord('q')):
        f.close()
        break
# Stop the camera
cam.release()
# Close all windows
cv2.destroyAllWindows()
#


































