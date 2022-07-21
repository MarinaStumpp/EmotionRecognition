For Speech Emotion recognition

- Training: Bash script speech_training.sh accesses the training process. Variables like epochs runs etc of arguement parser can be set. Standard values are set in the parser. Bash script in original case trains our model with 100 epochs for one run.

- Plotting: Bash script plot_speech_training.sh accesses the plotting process. It's possible to plot multiple runs. The variable file_path can be changed to access different plotting files.

--> Fast usage, just click on the bash scripts and they process the standard settings.


For Video Emotion recognition

- Execute the python file combined_demo
- For demo purposes the video can be changed to one of the videos available in the demo_videos folder.
- To plot the confusion matrix and test the combined model on the test dataset run the combined_test file.


Required packages:
- tensorflow
- numpy 
- matplotlib
- sklearn
- pandas
- librosa
- seaborn
- pyAudioAnalysis
- moviepy