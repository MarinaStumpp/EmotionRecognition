from video_prediction import process_video
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('combined-model.h5')
CATEGORIES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


# load video
video = "C:/Users/Philipp/Desktop/ER-Projekt/org-data/Actor_07/01-01-03-02-01-01-07.mp4"


# process video
video_predictions = process_video(video)


# process audio
audio_predictions = # call audio predict function


# combine video and audio predictions
X = video_predictions + audio_predictions


# use combined model for prediction
predictions = model.predict(X)


# mean of list is the index of the category that was predicted in total
print(f'Emotion Detected: {CATEGORIES[int(np.argmax(predictions))]}')
