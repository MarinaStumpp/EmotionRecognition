import argparse
from speech_emotion_recognition import SpeechEmotion
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # speech emotion recognition parameters
    parser.add_argument('--speech_data_available', type=str, default=False) # True or false, is speech feature data available
    parser.add_argument('--speech_feature_type', type=str, default='mfcc') # mfcc or all
    parser.add_argument('--speech_model_type', type=str, default='our_model') # original, original_lstm, original_cnn, our_model
    parser.add_argument('--speech_epochs', type=str, default=10) # number of training epochs
    parser.add_argument('--speech_runs', type=str, default=1) # number of runs to collect data for plotting
    parser.add_argument('--speech_ravdess_path', type=str, default='ravdess-emotional-speech-audio') # path to data
    parser.add_argument('--speech_audio_pickle_path', type=str, default='plotting_data/audio.dat') # path where plotting data is saved
    parser.add_argument('--speech_isConfusion', type=str, default='True') # Output of one confusion matrix
    parser.add_argument('--video_path', type=str, default='ravdess-emotional-speech-audio/Actor_01/02-01-08-02-02-02-01.mp4') # Output of one confusion matrix
    parser.add_argument('--audio_path', type=str, default='audio') # Output of one confusion matrix
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


if __name__ == "__main__":
    
    args = parse_args()
    speech_model = SpeechEmotion(**args)
    speech_model(**args)
    print(speech_model.process_audio(**args))