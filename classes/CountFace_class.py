import cv2 
import numpy as np 
import dlib

class count_faces():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.face_haarcascade = cv2.CascadeClassifier('./classes/model/haarcascade_frontalface_default.xml')

    def count_the_number_of_faces_haarcascade(self,img):
        faces = self.face_haarcascade.detectMultiScale(img,scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
        return len(faces)

    def count_the_number_of_faces_dlib(self,img):
        faces = self.detector(img)
        return len(faces)