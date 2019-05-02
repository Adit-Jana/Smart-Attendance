import os
import cv2
import numpy as np
from PIL import Image

#recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer=cv2.face.LBPHFaceRecognizer_create()
path = 'DataSet'

def getImagesandId(path):
    imagePaths=[os.path.join(path,d) for d in os.listdir(path)]
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("train",faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces

Ids,faces= getImagesandId(path)
recognizer.train(faces,Ids)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()


