# _D=False
# _C='cell phone'
# _B='tvmonitor'
# _A='laptop'
# import cv2,numpy as np
# class object_detect:
# 	def __init__(A):
# 		B='remote';A.object_name=['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant',_A,B,_B,_C];A.net=cv2.dnn.readNetFromCaffe('.\\classes\\model\\MobileNetSSD_deploy.prototxt.txt','.\\classes\\model\\MobileNetSSD_deploy.caffemodel');A.net_yolo=cv2.dnn.readNetFromDarknet('.\\classes\\model\\yolov4-tiny.cfg','.\\classes\\model\\yolov4-tiny.weights')
# 		with open('.\\classes\\model\\coco.names.txt')as C:A.classes=C.read().split('\n')
# 		A.classes_to_look_for=[_C,_A,B,_B];A.layer_names=A.net_yolo.getLayerNames();A.out_layers_indexes=A.net_yolo.getUnconnectedOutLayers();A.out_layers=[A.layer_names[B-1]for B in A.out_layers_indexes]
# 	def cell_phone_detector(A,image):
# 		D=image;I,J=D.shape[:2];G=cv2.dnn.blobFromImage(cv2.resize(D,(300,300)),.007843,(300,300),127.5);A.net.setInput(G);B=A.net.forward()
# 		for E in np.arange(0,B.shape[2]):
# 			F=B[(0,0,E,2)]
# 			if F>.5:
# 				C=int(B[(0,0,E,1)])
# 				if str(A.object_name[C])==_C or str(A.object_name[C])==_A or str(A.object_name[C])==_B:
# 					H=.8
# 					if F>=H:return True
# 		return _D
# 	def cell_phone_detector_second(A,image):
# 		F=image;G,H,W=F.shape;P=cv2.dnn.blobFromImage(F,1/255,(608,608),(0,0,0),swapRB=True,crop=_D);A.net_yolo.setInput(P);Q=A.net_yolo.forward(A.out_layers);I,J,E=([]for A in range(3));R=0
# 		for S in Q:
# 			for B in S:
# 				K=B[5:];C=np.argmax(K);L=K[C]
# 				if L>0:T=int(B[0]*H);U=int(B[1]*G);M=int(B[2]*H);N=int(B[3]*G);O=[T-M//2,U-N//2,M,N];E.append(O);I.append(C);J.append(float(L))
# 		V=cv2.dnn.NMSBoxes(E,J,.0,.4)
# 		for D in V:
# 			D=D;O=E[D];C=I[D]
# 			if A.classes[C]in A.classes_to_look_for:R+=1;return True
# 		return _D
	
import cv2
import numpy as np

class object_detect():

    def __init__(self):
        self.object_name = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "laptop",
        "remote", "tvmonitor", "cell phone"]
        self.net = cv2.dnn.readNetFromCaffe(".\classes\model\MobileNetSSD_deploy.prototxt.txt", ".\classes\model\MobileNetSSD_deploy.caffemodel" )
        self.net_yolo = cv2.dnn.readNetFromDarknet( ".\classes\model\yolov4-tiny.cfg" , ".\classes\model\yolov4-tiny.weights" )
        with open(".\classes\model\coco.names.txt") as file:
            self.classes = file.read().split("\n")
        self.classes_to_look_for = ["cell phone","laptop","remote","tvmonitor"]
        self.layer_names = self.net_yolo.getLayerNames()
        self.out_layers_indexes = self.net_yolo.getUnconnectedOutLayers()
        self.out_layers = [self.layer_names[index - 1] for index in self.out_layers_indexes]


    def cell_phone_detector(self,image):

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])

                if str(self.object_name[idx]) == "cell phone" or str(self.object_name[idx]) == "laptop" or str(self.object_name[idx]) == "tvmonitor":
                    threshold = 0.8
                    if confidence >= threshold:
                        return True
                    
        return False
    

    def cell_phone_detector_second(self,image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
        self.net_yolo.setInput(blob)
        outs = self.net_yolo.forward(self.out_layers)
        class_indexes, class_scores, boxes = ([] for i in range(3))
        objects_count = 0

        for out in outs:
            for obj in out:
                scores = obj[5:]
                class_index = np.argmax(scores)
                class_score = scores[class_index]
                if class_score > 0:
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    obj_width = int(obj[2] * width)
                    obj_height = int(obj[3] * height)
                    box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                    boxes.append(box)
                    class_indexes.append(class_index)
                    class_scores.append(float(class_score))

        chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
        for box_index in chosen_boxes:
            box_index = box_index
            box = boxes[box_index]
            class_index = class_indexes[box_index]

            if self.classes[class_index] in self.classes_to_look_for:
                objects_count += 1
                return True

        return False 