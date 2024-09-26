from imutils.video import VideoStream
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2

class liveness_detection():
    def __init__(self):
        self.protoPath =  r".\classes\model\deploy.prototxt"
        self.modelPath =  r".\classes\model\res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        self.model = load_model("./classes/model/liveness.model")
        self.le = pickle.loads(open("./classes/model/le.pickle", "rb").read())
        
    def detector(self,img):
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        label = "None"
        preds = [0.0]
        j = 0
        pred_value = 0.0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.85:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                face = img[startY:endY, startX:endX]
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)
                preds = self.model.predict(face)[0]
                j = np.argmax(preds)
                label = self.le.classes_[j]
                pred_value = preds[j]
            
            return label,pred_value