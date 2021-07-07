#sebenernya ini adalah tracking muka wkwkwk
#videocapture with detect faces
import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#img = cv2.imread("///Users/macbookpro/Documents/gua19.jpeg")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow
#to capture video from webcam
webcam = cv2.VideoCapture(0)
#Iterate forever over frames
while True:
    successful_frame_read, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)
    cv2.imshow("original", frame)
    cv2.waitKey(1)





#face_coordinates = trained_face_data.detectMultiScale(gray)
#cv2.rectangle(img, (160, 377), (160+257, 377+257), (0, 255, 0), 2)
#print(face_coordinates)
#cv2.imshow("image original", img)
#cv2.waitKey(0)