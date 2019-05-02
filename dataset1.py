import cv2
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
Id=input('enter id: ')
Name=input('enter name: ')
#data base handelling
def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM PEOPLE WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    #flag
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE people SET NAME="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO people(Id,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

insertOrUpdate(Id,Name)
    
sampleNum=0;
while(True):
    ret,img = cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    
    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("DataSet/User."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2);
        cv2.waitKey(100);
    cv2.imshow("Face",img);
    cv2.waitKey(1);
    if(sampleNum>20):
        break;
cam.release()
cv2.destroyAllWindows()








































