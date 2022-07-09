import pandas as pd
import numpy as np
from video_prediction import process_video


# create an empty dataframe
df = pd.DataFrame()


# for each video file


# process video
video_prediction = process_video(video_path)


# process audio
audio_prediction = # add audio prediction


# combine the outputs
predictions = video_prediction + audio_prediction


# append the data and the label to the csv file


# store the csv file
df.to_csv('predictions.csv', index=False)