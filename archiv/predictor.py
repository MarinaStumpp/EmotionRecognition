# imports
import numpy as np
import cv2
import tensorflow as tf
import statistics

video_dir = "C:/Users/Philipp/Desktop/ER-Projekt/org-data/Actor_07/01-01-03-02-01-01-07.mp4"
IMG_SIZE = 300
loaded_images = []
processed_images = []
model = tf.keras.models.load_model('model.h5')
CATEGORIES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def process_video():

    # get all images from mp4
    vidcap = cv2.VideoCapture(video_dir)
    success, image = vidcap.read()
    print('Starting Video Processing...')
    while success:
        success, image = vidcap.read()
        loaded_images.append(image)
    print('Video loaded')
    print('Starting Image Generation...')

    # values for info output
    no_face_count = 0
    img_count = 0

    # image classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # iterate all images in the folder
    for img in loaded_images:

        # info output
        img_count += 1

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
                no_face_count += 1
            else:

                # resize and append to training data
                new_array = cv2.resize(face_crop[0], (IMG_SIZE, IMG_SIZE))

                processed_images.append(new_array)  # add this to our training_data
        except Exception as e:
            print("general exception", e)

    print(f'Total images processed: {img_count} - Images with no face recognized: {no_face_count}')


# processing of the video
process_video()

# reshape
X = np.array(processed_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# predict all the images with the model
print('Finished Image Processing')
print('Starting Predictions...')
predictions = model.predict(X)

# add the prediction index to the list
new_predictions = []
for prediction in predictions:
    new_predictions.append(np.argmax(prediction))

# mean of list is the index of the category that was predicted in total
mean = statistics.mean(new_predictions)
print(f'Emotion Detected: {CATEGORIES[int(mean)]}')
