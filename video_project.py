import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

train_path = 'CSE_438_dataset'


def detect_face(input_img):
    image = cv.cvtColor(input_img,
                        cv.COLOR_BGR2GRAY)  # cv2.cvtColor() method is used to convert an image from one color space to another.
    face_cascade = cv.CascadeClassifier(
        "haarcascade_frontalface_default.xml")  # Object Detection using Haar feature-based cascade classifiers is an effective object detection method
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5);
    # print(faces)                       #[[ 46  53 114 114]] = 2D
    if (len(faces) == 0):
        return 0, 0
    (x, y, w, h) = faces[0]
    # print(faces[0])                        #[ 46  53 114 114] = 1D
    return image[y:y + w, x:x + h], faces[0]


def prepare_training_data(train_path):
    detected_faces = []
    face_labels = []
    traning_image_dirs = os.listdir(
        train_path)  # listdir() returns a list containing the names of the entries in the directory given by path
    for dir_name in traning_image_dirs:
        label = str(dir_name)
        training_image_path = train_path + "/" + dir_name
        training_images_names = os.listdir(training_image_path)

        for image_name in training_images_names:
            image_path = training_image_path + "/" + image_name
            image = cv.imread(image_path)
            face, rect = detect_face(image)

            if face is not 0:
                resized_face = cv.resize(face, (121, 121),
                                         interpolation=cv.INTER_AREA)  # Different interpolation methods are used for different resizing purposes.
                # INTER_AREA uses pixel area relation for resampling. This is best suited for reducing the size of an image (shrinking).
                detected_faces.append(face)
                face_labels.append(label)

    return detected_faces, face_labels


detected_faces, face_labels = prepare_training_data(train_path)


#print("Total faces: ", len(detected_faces))
#print("Total labels: ", len(face_labels))
#cv.imshow('h',detected_faces[10])

from sklearn import preprocessing
lebel = preprocessing.LabelEncoder()
int_lebel = lebel.fit_transform(face_labels)
int_lebel

#initialize a face recognizer
lbphfaces_recognizer = cv.face.LBPHFaceRecognizer_create()

#train the face recognizer model
lbphfaces_recognizer.train(detected_faces, np.array(int_lebel))

def draw_rectangle(test_image, rect):
    (x, y, w, h) = rect
    cv.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)#x1y1 ; x2y2; color; thickness

def draw_text(test_image, label_text, x, y):
    cv.putText(test_image, label_text, (x, y), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)# 1.5=thickness


