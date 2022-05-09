import cv2
from random import randrange

#load pre-trained data from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#image import
img = cv2.imread('')#image path inside ''

#convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Face detection
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#face square
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)

#img display
cv2.imshow("Face detector", img)
cv2.waitKey()

print("Code completed")