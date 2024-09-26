import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

class eye_blink_detector():

    def __init__(self):
        self.blink_thresh = 0.26
        self.succ_frame = 2
        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predict= dlib.shape_predictor('./classes/model/shape_predictor_68_face_landmarks.dat')

    def calculate_EAR(self,eye): 
        y1 = dist.euclidean(eye[1], eye[5]) 
        y2 = dist.euclidean(eye[2], eye[4])    
        x1 = dist.euclidean(eye[0], eye[3]) 
        EAR = (y1+y2) / (x1*2.0) 
        return EAR
    
    def eye_blink(self,img_gray,face,count_frame,total):
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

        shape = self.landmark_predict(img_gray, face) 
        shape = face_utils.shape_to_np(shape) 
 
        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 

        left_EAR = self.calculate_EAR(lefteye) 
        right_EAR = self.calculate_EAR(righteye) 
        avg = (left_EAR+right_EAR)/2.0

        if avg < self.blink_thresh: 
            count_frame += 1 
        else: 
            if count_frame >= self.succ_frame:
                total+=1
            count_frame = 0
        
        return count_frame,total