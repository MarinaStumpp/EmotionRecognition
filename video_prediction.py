# imports
import numpy as np
import cv2
import tensorflow as tf
import statistics


def process_video(video_dir):

    IMG_SIZE = 300
    loaded_images = []
    processed_images = []
    model = tf.keras.models.load_model('vgg16-model.h5')
    CATEGORIES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    # get all images from mp4
    vidcap = cv2.VideoCapture(video_dir)
    success, image = vidcap.read()
    while success:
        success, image = vidcap.read()
        loaded_images.append(image)

    # image classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # iterate all images in the folder
    for img in loaded_images:

        try:

            # read the image
            img_array = img

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
            else:
                # resize and append to training data
                new_array = cv2.resize(face_crop[0], (IMG_SIZE, IMG_SIZE))

                # add this to our training_data
                processed_images.append(new_array)
        except Exception as e:
            print("general exception", e)

    # reshape
    X = np.array(processed_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # predict all the images with the model
    predictions = model.predict(X)

    # create an array with the means of the predictions
    means = []
    for count, cat in enumerate(CATEGORIES):
        means.append(float(statistics.mean([el[count] for el in predictions])))

    return means
