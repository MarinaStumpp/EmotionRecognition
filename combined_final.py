from video_prediction import process_video
import tensorflow as tf
import numpy as np
from speech_emotion_recognition import SpeechEmotion
import argparse


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # speech emotion recognition parameters
    parser.add_argument('--speech_data_available', type=str, default=False) # True or false, is speech feature data available
    parser.add_argument('--speech_feature_type', type=str, default='mfcc') # mfcc or all
    parser.add_argument('--speech_model_type', type=str, default='our_model') # original, original_lstm, original_cnn, our_model
    parser.add_argument('--speech_epochs', type=str, default=100) # number of training epochs
    parser.add_argument('--speech_runs', type=str, default=1) # number of runs to collect data for plotting
    parser.add_argument('--speech_ravdess_path', type=str, default='speech_files') # path to data
    parser.add_argument('--speech_audio_pickle_path', type=str, default='plotting_data/audio.dat') # path where plotting data is saved
    parser.add_argument('--speech_isConfusion', type=str, default=False) # Output of one confusion matrix
    parser.add_argument('--audio_path', type=str, default='speech_files/tmp/audio.wav') # tmp file path for audio
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


if __name__ == "__main__":
    args = parse_args()
    model = tf.keras.models.load_model('combined-model.h5')
    CATEGORIES = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
    speech_emotion = SpeechEmotion(**args)

    # load video
    #video = "C:/Users/Philipp/Desktop/ER-Projekt/training_data/sad/01-01-04-01-01-01-01.mp4"
    video = "C:/Users/Philipp/Desktop/test.mp4"

    # process video
    video_prediction = process_video(video)

    # process audio
    audio_prediction = speech_emotion.process_audio(video, **args)
    #audio_prediction = video_prediction

    # combine video and audio predictions
    predictions = np.append(video_prediction, audio_prediction)
    predictions = np.reshape(predictions, (1, 16))

    # use combined model for prediction
    predictions = model.predict(predictions)

    print(predictions)
    # mean of list is the index of the category that was predicted in total
    print(f'Emotion Detected: {CATEGORIES[int(np.argmax(predictions))]}')
