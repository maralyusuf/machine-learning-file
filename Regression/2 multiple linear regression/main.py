import cv2
import numpy as np
import dlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def middle(p1,p2):
    return int((p1[0] + p2[0]) / 2),int((p1[1]+ p2[1])/ 2)


file = open("dataset.csv","a")

dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:,:2].values
y = dataset.iloc[:,2:].values


liner = LinearRegression()

liner.fit(x,y)


while True:
     _,frame = cap.read()

     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

     faces = detector(gray)
     for face in faces:

        points = model(gray,face)

        face_p = [(p.x,p.y) for p in points.parts()]

        face_dict = {"left":face_p[42:48],"right":face_p[36:42],"nose":face_p[27:31]}

        ## goz un kapalı olup olmadıgı
        up = middle(face_dict["left"][1],face_dict["left"][2])
        down = middle(face_dict["left"][4],face_dict["left"][5])
        cv2.line(frame,up,down,(0,0,255),2)
        goz = (down[1] - up[1])

        p1,p2 = face_dict["nose"][0],face_dict["nose"][3]

        nose = p2[1]-p1[1]
        print("deger : ",nose)

        pred = liner.predict([[nose,goz]])

        if pred >= 0.5:
            pred = 1
        else:
            pred = 0

        cv2.putText(frame,str(pred),up,cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)




     cv2.imshow("frame",frame)

     if cv2.waitKey(25) & 0xFF == ord("q"):
        print("nose : ",nose," eye : ",goz)
        a = input("deger : ")

        if a =="q":
         break
        elif a == "k":
            file.write(str(nose)+","+str(goz)+",0\n")

        elif a == "a":
            file.write(str(nose)+","+str(goz)+",1\n")


cap.release()
cv2.destroyAllWindows()
