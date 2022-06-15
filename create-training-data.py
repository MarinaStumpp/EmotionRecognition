import numpy as np
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/Philipp/Desktop/ER-Projekt/data"

CATEGORIES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

IMG_SIZE = 300

training_data = []

'''
for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR, category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        img_array = cv2.imread(os.path.join(path, img))

        # Convert into grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        face_crop = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_crop.append(gray[y:y + h, x:x + w])

        # Display the output
        for face in face_crop:
            print(face_crop)

        new_array = cv2.resize(face_crop[0], (IMG_SIZE, IMG_SIZE))
        cv2.imshow('img', new_array)
        cv2.waitKey()

        break  # we just want one for now so break
    break  # ...and one more!
'''


def create_training_data():

    # values for info output
    no_face_count = 0
    img_count = 0

    # image classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # iterate all categories
    for category in CATEGORIES:

        # create the path and get the label
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        # iterate all images in the folder
        for img in os.listdir(path):

            # info output
            img_count += 1
            print(f'category: {category} - img_total: {img_count} - no_face: {no_face_count}')

            try:

                # read the image
                img_array = cv2.imread(os.path.join(path, img))

                # convert into grayscale
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                # detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                # draw rectangle around the faces and crop the face out of the image
                face_crop = []
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_crop.append(gray[y:y + h, x:x + w])

                # if there is no face or multiple found skip the image
                if len(face_crop) != 1:
                    print('no face found')
                    no_face_count += 1
                else:

                    # resize and append to training data
                    new_array = cv2.resize(face_crop[0], (IMG_SIZE, IMG_SIZE))

                    training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:
                print("general exception", e, os.path.join(path, img))


create_training_data()

print(len(training_data))

# shuffle training data for randomness
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# safe data to pickle files
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
