import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import video_project as fr
print (fr)


# predict output on test data
def predict(test_image):
    face, rect = fr.detect_face(test_image)
    label, confidence = fr.lbphfaces_recognizer.predict(face)
    tags ={11: "2019-1-60-170", 9: "2019-1-60-094", 12: "2019-1-60-171", 14: "2019-1-60-173",
                   15: "2019-1-60-174", 7: "2019-1-60-075", 8: "2019-1-60-093",
                   16: "2019-1-60-204", 10: "2019-1-60-098", 13: "2019-1-60-172", 5: "2019-1-60-060",
                   3: "2019-1-60-024", 4: "2019-1-60-055", 2: "2019-1-60-005",
                   6: "2019-1-60-066", 1: "2018-2-60-048", 0: "2018-2-60-046"}

    print(label)  # (3, 62.21555307530192)
    label_text = tags[label]
    # label_text = name[label]
    # label_text = label[0]
    fr.draw_rectangle(test_image, rect)
    fr.draw_text(test_image, label_text, rect[0], rect[1] - 5)  # -5 from y
    return test_image, label_text, confidence




predicted_image, label_id, confidence = predict(cv2.imread(r'CSE_438_dataset\faria\2.jpg'))


test_img = cv2.imread(r'CSE_438_dataset\faria\2.jpg')#Give path to the image which you want to test

fig = plt.figure()
ax1 = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax1.set_title('confidence: ' + str(confidence) + ' | ' + 'predicted class: ' + label_id)
plt.axis("off")
imgplot = plt.imshow(cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
plt.show()

cv2.waitKey(0)