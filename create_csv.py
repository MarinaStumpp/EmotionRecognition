import pandas as pd
import numpy as np
from video_prediction import process_video
import os
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
    parser.add_argument('--speech_ravdess_path', type=str, default='ravdess-emotional-speech-audio') # path to data
    parser.add_argument('--speech_audio_pickle_path', type=str, default='plotting_data/audio.dat') # path where plotting data is saved
    parser.add_argument('--speech_isConfusion', type=str, default='True') # Output of one confusion matrix
    # parser.add_argument('--video_path', type=str, default='ravdess-emotional-speech-audio/Actor_01/02-01-08-02-02-02-01.mp4') # Output of one confusion matrix
    parser.add_argument('--audio_path', type=str, default='audio') # Output of one confusion matrix
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


if __name__ == "__main__":

    args = parse_args()
    # create an empty dataframe
    df = pd.DataFrame()
    speech_recognition = SpeechEmotion(**args)


    # for each video file
    subfolders = [f.path for f in os.scandir('videos') if f.is_dir()]

    for index, folder in enumerate(subfolders):

        # iterate all files in folder
        for filename in os.listdir(folder):
            video_path = os.path.join(folder, filename)

            # checking if it is a file
            if os.path.isfile(video_path):

                # process video
                video_prediction = process_video(video_path)

                # process audio
                audio_prediction = speech_recognition.process_audio(video_path, **args)

                # combine the outputs
                predictions = np.append(video_prediction, audio_prediction)

                # append the label
                predictions = np.append(predictions, index)

                # reshape the array
                predictions = np.reshape(predictions, (1, 17))

                # append the data and the label to the csv file
                new_data = pd.DataFrame(predictions, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 'label'])

                # append the new data to the dataframe
                df = pd.concat([df, new_data], ignore_index=True)

    # store the csv file
    df.to_csv('predictions.csv', index=False)
