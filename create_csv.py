import pandas as pd
import numpy as np
from video_prediction import process_video
import os


# create an empty dataframe
df = pd.DataFrame()


# for each video file
subfolders = [f.path for f in os.scandir('videos') if f.is_dir()]

for index, folder in enumerate(subfolders):

    # iterate all files in folder
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)

        # checking if it is a file
        if os.path.isfile(f):

            # process video
            video_prediction = process_video(f)

            # process audio
            # TODO change to process_audio()
            audio_prediction = video_prediction

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
