import pandas as pd
import numpy as np
from video_prediction import process_video


# create an empty dataframe
df = pd.DataFrame()


# for each video file
videos = [
    "C:/Users/Philipp/Desktop/ER-Projekt/training_data/angry/01-01-05-01-01-01-01.mp4",
    "C:/Users/Philipp/Desktop/ER-Projekt/training_data/angry/01-01-05-01-01-01-02.mp4",
    "C:/Users/Philipp/Desktop/ER-Projekt/training_data/angry/01-01-05-01-01-01-03.mp4",
    "C:/Users/Philipp/Desktop/ER-Projekt/training_data/angry/01-01-05-01-01-01-04.mp4",
    "C:/Users/Philipp/Desktop/ER-Projekt/training_data/angry/01-01-05-01-01-01-05.mp4"
]

# iterate over all videos
for video_dir in videos:

    # process video
    video_prediction = process_video(video_dir)

    # process audio
    # TODO change to process_audio()
    audio_prediction = process_video(video_dir)

    # combine the outputs
    predictions = np.append(video_prediction, audio_prediction)

    # append the label
    predictions = np.append(predictions, 0)

    # reshape the array
    predictions = np.reshape(predictions, (1, 17))

    # append the data and the label to the csv file
    new_data = pd.DataFrame(predictions, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 'label'])

    # append the new data to the dataframe
    df = pd.concat([df, new_data], ignore_index=True)

# store the csv file
df.to_csv('predictions.csv', index=False)
