from video_prediction import process_video
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('combined-model.h5')
CATEGORIES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


# load video
video = "C:/Users/Philipp/Desktop/ER-Projekt/training_data/angry/01-01-05-01-01-01-05.mp4"


# process video
video_prediction = process_video(video)


# process audio
# TODO change to process_audio()
audio_prediction = process_video(video)


# combine video and audio predictions
predictions = np.append(video_prediction, audio_prediction)
predictions = np.reshape(predictions, (1, 16))


# use combined model for prediction
predictions = model.predict(predictions)


# mean of list is the index of the category that was predicted in total
print(f'Emotion Detected: {CATEGORIES[int(np.argmax(predictions))]}')
