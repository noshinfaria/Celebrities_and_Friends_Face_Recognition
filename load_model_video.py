import numpy as np
import cv2 as cv
import os

import video_project as fr
print (fr)


cap = cv.VideoCapture(0)  # If you want to recognise face from a video then replace 0 with video path
while True:
    ret, test_img = cap.read()

    faces_detected, gray_img = fr.detect_face(test_img)
    print("face Detected: ", faces_detected)

    for (x, y, w, h) in faces_detected:
        cv.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)

    for face in faces_detected:
        (x, y, w, h) = face
        roi_gray = gray_img[y:y + h, x:x + h]
        label, confidence = fr.lbphfaces_recognizer.predict(roi_gray)
        name = {11: "2019-1-60-170", 9: "2019-1-60-094", 12: "2019-1-60-171", 14: "2019-1-60-173",
                       15: "2019-1-60-174", 7: "2019-1-60-075", 8: "2019-1-60-093",
                       16: "2019-1-60-204", 10: "2019-1-60-098", 13: "2019-1-60-172", 5: "2019-1-60-060",
                       3: "2019-1-60-024", 4: "2019-1-60-055", 2: "2019-1-60-005",
                       6: "2019-1-60-066", 1: "2018-2-60-048", 0: "2018-2-60-046", 17: "2018-2-60-041",
                        18: "2018-2-60-046", 19: "2018-2-60-046", 20: "2018-2-60-046", 21: "2018-2-60-046",
                        22: "2018-2-60-046", 23: "2018-2-60-046", 24: "2018-2-60-046", 25: "2018-2-60-046",
                        26: "2018-2-60-046", 27: "2018-2-60-046", 28: "2018-2-60-046", 29: "2018-2-60-046"}
        print("Confidence :", confidence)
        print("label :", label)
        fr.draw_rectangle(test_img, face)
        predicted_name = name[label]
        fr.draw_text(test_img, predicted_name, x, y)

    resized_img = cv.resize(test_img, (1000, 700))

    cv.imshow("face detection ", resized_img)
    if cv.waitKey(10) == ord('q'):
        break

cv.waitKey(0)