import cv2
import dlib
import scipy.spatial.distance as dist 
from imutils import face_utils
import numpy as np

from classes import blink_class
from classes import orientation_class
from classes import liveness_class
from classes import count_class

from imutils.video import VideoStream
import time

cap = VideoStream(src=1).start()

real_detected_time = 0
penalty_time = 0

type_detect=""
procent_detect=0

count_faces = 0
count_frame=0
total=0

test1=liveness_class.liveness_detection()
test2=blink_class.eye_blink_detector()
test3=orientation_class.face_orientation()
test_bonus = count_class.count_faces()

step1=True
step2=False
step3=False

count_first = 0
count_second = 0

real_detect = False

while True:

    img = cap.read()
    img = cv2.flip(img,1)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    count_first = test_bonus.count_the_number_of_faces_haarcascade(img_gray)
    count_second = test_bonus.count_the_number_of_faces_dlib(img_gray)

    if (count_first >= 2 and count_second >=2) or ((count_first == 0 and count_second == 0)):
        step1 = False
        step2 = False
        step3 = False
        penalty_time = time.time()
    elif ((time.time() - penalty_time) > 3.0) and real_detect==False:
        print("Time has run out")
        step1 = True

    if step1:
        step2 = False
        step3 = False
        type_detect,procent_detect = test1.detector(img)
        if (type_detect == "real") and (procent_detect >= 0.9):
            if real_detected_time == 0:
                real_detected_time = time.time()
            elif (time.time() - real_detected_time) > 5.0:
                step1 = False
                step2 = True
                real_detect = True
                print("real face")
        elif (type_detect == "spoof"):
            print("spoof face")
            real_detected_time = 0
            step1 = True
            real_detect = False
        else:
            print("unknown face")
            real_detected_time = 0

    if step2:
        step1 = False
        step3 = False
        faces = test2.detector(img_gray)
        for face in faces:
            count_frame,total= test2.eye_blink(img_gray,face,count_frame,total)
            if total >= 4:
                print("blink detect")
                step2=False
                step3=True
                total=0
                count_frame=0
    

    # if next1:
    #     boxes,names = test2.face_orientation(img_gray)
    #     img=test2.bounding_box(img,boxes,names)
    #     print(names)
    #     if "right" in names:
    #         print("right")

    # img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    cv2.imshow("image",img)
    if cv2.waitKey(5) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows() 

# Первым идёт проверка на блики
# Вторым моргание
# Третьем повороты головы
