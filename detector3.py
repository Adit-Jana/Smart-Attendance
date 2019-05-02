#import open cv for image processing
import cv2
import numpy as np
import os


# Detect object in video stream using Haarcascade Frontal Face
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer\\trainingData.yml')

#here give font and color of the text
font=cv2.FONT_HERSHEY_SIMPLEX
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
        cv2.rectangle(im,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),4)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="adit"
            
        #cv2.putText(img,str(id),(x,y+h),font,255)
        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(id), (x,y-40), font, 1, (255,255,255), 3)
    # Display the video frame with the bounded rectangle    
    cv2.imshow("Face",im)
    # If 'q' is pressed, close program
    if(cv2.waitKey(1)==ord('q')):
        break
# Stop the camera
cam.release()
# Close all windows
cv2.destroyAllWindows()






































