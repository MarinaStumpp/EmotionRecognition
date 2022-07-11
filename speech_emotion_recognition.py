# Basic imports
from email.mime import audio
import pandas as pd
import numpy as np
import os
import sys
import csv

# librosa is a Python library for analyzing audio and music
import librosa
import librosa.display
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

#sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

#Tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import regularizers

# Py audio analysis
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

# extract audio from mp4
from moviepy.editor import VideoFileClip


class SpeechEmotion(object):

    def __init__(self, speech_data_available, speech_feature_type, speech_epochs, speech_runs, speech_audio_pickle_path, speech_isConfusion, speech_ravdess_path, **kwargs):
        self.pickle_path = speech_audio_pickle_path
        self.data_available = bool(speech_data_available)
        self.feature_type = speech_feature_type
        self.epochs = int(speech_epochs)
        self.runs = int(speech_runs)
        self.isConfusion = bool(speech_isConfusion)
        self.ravdess_path = speech_ravdess_path
        
    def __call__(self, speech_model_type, **kwargs):
        self.model = self.train_model(speech_model_type, **kwargs)
        return self.model
    
    def process_audio(self, video_path, audio_path, **kwargs):
        audio_features=[]
        audio_path=self.convert_video_to_audio(video_path, audio_path, **kwargs)
        if self.feature_type=='mfcc':
            audio_features.append(self.extract_mfcc_features(audio_path))
        else:
            audio_features.append(self.extract_features(audio_path))
        audio_features = np.stack(audio_features)
        scaler = StandardScaler()
        audio_features = scaler.fit_transform(audio_features.reshape(-1, audio_features.shape[-1])).reshape(audio_features.shape)
        return self.model.predict(audio_features)



    # Create model and train
    def train_model(self, speech_model_type, **kwargs):
        plotting_data=[]
        for i in range(self.runs):
            x_train, x_test, y_train, y_test, encoder = self.prepare_training_data(self.data_available, self.feature_type)
            self.data_available = True
            model = self.create_model(speech_model_type, x_train.shape[1:])
            opt = keras.optimizers.Adam()
            model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
            model.summary()
            rlrp = ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=1, patience=5, min_lr=0.0000001)
            #earlyStopping = EarlyStopping(monitor ="val_accuracy", mode = 'auto', patience =10, restore_best_weights = True)
            history=model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=64, epochs=self.epochs, callbacks=[rlrp], shuffle=True)
            plotting_data.append(history.history)

        model.save('speech-model.h5')

        with open(self.pickle_path, "wb") as f:
            pickle.dump((plotting_data), f)

        if(self.isConfusion):
            self.plot_confusion_matrix(model, encoder, x_test, y_test, speech_model_type, **kwargs)
        return model
    
    #Convert the mp4 to wav file
    def convert_video_to_audio(self, video_file, audio_file_location, out_ext="wav", **kwargs):
        filename, ext = os.path.splitext(video_file)
        clip = VideoFileClip(video_file)
        clip.audio.write_audiofile(f"{audio_file_location.split('.')[0]}.{out_ext}", codec='pcm_s16le')
        return (f"{audio_file_location.split('.')[0]}.{out_ext}")

    # Get paths for data from mp4 data to mp3 data
    def preprocess_ravdess_data(self, **kwargs):
        ravdess_directory_list=os.listdir(self.ravdess_path)
        file_emotion = []
        file_path = []
        for path in ravdess_directory_list:
            actor = os.listdir(self.ravdess_path +'/' +path)
            for file in actor:
                part = file.split('.')[0]
                part = part.split('-')
                # third part in each file represents the emotion associated to that file.
                file_emotion.append(int(part[2]))
                dir = 'speech_files'+'/'+ path +'/'+ file.split('.')[0] + '.wav'
                if not os.path.exists(dir):
                    dir = self.convert_video_to_audio(self.ravdess_path +'/'+ path +'/'+ file, 'speech_files'+'/'+ path +'/'+ file, out_ext='wav', **kwargs)
                file_path.append(dir)

        # dataframe for emotion of files
        emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

        # dataframe for path of files.
        path_df = pd.DataFrame(file_path, columns=['Path'])
        data_path = pd.concat([emotion_df, path_df], axis=1)

        # changing integers to actual emotions.
        data_path.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
        data_path.to_csv("feature_data/data_path.csv",index=False)
        data_path.head()
        
        return data_path

    # extract short term features with py audioanalysis
    def extract_short_term_features(self, path, **kwargs):
        [Fs, data] = audioBasicIO.read_audio_file(path)# read the wav file
        data = audioBasicIO.stereo_to_mono(data) 
        results, feature_names = ShortTermFeatures.feature_extraction(data, Fs, 0.050*Fs, 0.025*Fs, deltas=False)
        if results.shape[1]>100:
            results=results[:,:100]
        elif results.shape[1]<100:
            padding = np.zeros((34,100))
            padding[:results.shape[0],:results.shape[1]]=results
            results = padding
        results = np.transpose(results)
        return results

    # MFCC features using librosa
    def extract_mfcc_features(self, path, **kwargs): 
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        results=[]

        # MFCC extraction
        results = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T
        
        if results.shape[0]>100:
            results=results[:100,:]
        elif results.shape[0]<100:
            padding = np.zeros((100,13))
            padding[:results.shape[0],:results.shape[1]]=results
            results=padding
            
        return results

    # get all features of all files by using py audio
    def extract_features(self,data_path, feature_type, **kwargs):
        features, labels = [], []
        if not os.path.exists('feature_data/'+feature_type+'_features.npy') and not os.path.exists('_feature_data/'+ feature_type +'_lables.npy'):

            for path, emotion in zip(data_path.Path, data_path.Emotions):
                if feature_type == 'all': 
                    #34 features
                    self.extract_short_term_features(path, **kwargs)
                elif feature_type == 'mfcc':
                    #mfcc
                    feature=self.extract_mfcc_features(path, **kwargs)
                else:
                    print("Feature type not available")
                features.append(feature)
                labels.append(emotion)
            np.save('feature_data/'+feature_type+'_features.npy', features)
            np.save('feature_data/'+feature_type+'_labels.npy', labels)
        else:
            features, labels = self.load_data_and_features(feature_type,**kwargs)
        
        return features, labels

    ## Read the data from database files
    def load_data_and_features(self, feature_type, **kwargs):
        features = np.load('feature_data/'+feature_type+'_features.npy')
        labels = np.load('feature_data/'+feature_type+'_labels.npy')
        return features, labels

    # Either get data from files or calculate the needed features
    def prepare_training_data(self, data_available, feature_type, **kwargs):
        # load data or generate features out of the RAVDESS database
        if data_available:
            features,labels= self.load_data_and_features(feature_type, **kwargs)
        else:
            data_path = self.preprocess_ravdess_data()
            features, labels = self.extract_features(data_path, feature_type, **kwargs)
        X = np.stack(features)
        Y = labels
        #One hot encoding the labels
        encoder = OneHotEncoder()
        Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
        # splitting data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True, test_size=0.2)
        # scaling data with sklearn's Standard scaler
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
        x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    
        return x_train, x_test, y_train, y_test, encoder

    # Original model used in the original paper
    def create_original_speech_model(self, input_shape, **kwargs):
        inputs = keras.Input(shape=(input_shape))

        #Low level features
        low_level = layers.Bidirectional(LSTM(256, return_sequences=True))(inputs)
        low_level = layers.Dropout(0.2)(low_level)
        low_level = layers.Bidirectional(LSTM(256))(low_level)
        low_level = layers.Dropout(0.2)(low_level)
        low_level = layers.Dense(8, activation='relu')(low_level)
        
        #High level features 
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=2, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.2)(high_level)
        high_level = layers.Flatten()(high_level)
        high_level = layers.Dense(8, activation='relu')(high_level)
        
        #Concatenate low and high level features
        merge = layers.concatenate([low_level, high_level], axis=1)
        merge = layers.Flatten()(merge)

        outputs = Dense(8, activation='softmax')(merge)

        model = keras.Model(inputs, outputs)
        return model

    # Original LSTM model
    def create_LSTM_speech_model(self, input_shape, **kwargs):
        inputs = keras.Input(shape=(input_shape))

        #Low level features
        low_level = layers.Bidirectional(LSTM(256, return_sequences=True))(inputs)
        low_level = layers.Dropout(0.2)(low_level)
        low_level = layers.Bidirectional(LSTM(256))(low_level)
        low_level = layers.Dropout(0.2)(low_level)
        outputs = layers.Dense(8, activation='softmax')(low_level)

        model = keras.Model(inputs, outputs)

        return model
    
    # Original CNN model
    def create_CNN_speech_model(self, input_shape, **kwargs):
        inputs = keras.Input(shape=(input_shape))
        
        #High level features 
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=2, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.2)(high_level)
        high_level = layers.Flatten()(high_level)

        outputs = layers.Dense(8, activation='softmax')(high_level)

        model = keras.Model(inputs, outputs)
        return model
    
    # Optimized Original Model
    def create_test_speech_model(self, input_shape, **kwargs):
        inputs = keras.Input(shape=(input_shape))

        #Low level feature
        low_level = layers.Bidirectional(LSTM(64, return_sequences=True))(inputs)
        low_level = layers.Dropout(0.5)(low_level)
        low_level = layers.Bidirectional(LSTM(32))(low_level)
        low_level = layers.Dropout(0.5)(low_level)
        low_level = layers.Dense(128)(low_level)
        low_level = layers.Dropout(0.5)(low_level)
        low_level = layers.Dense(64)(low_level)
        low_level = layers.Dropout(0.5)(low_level)
        
        #High level features 
        high_level = layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
        high_level = layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.4)(high_level)
        
        high_level = layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.3)(high_level)
        
        high_level = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.3)(high_level)
        
        high_level = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.3)(high_level)
        high_level = layers.Flatten()(high_level)

        #Concatenate low and high level features
        merge = layers.concatenate([low_level, high_level], axis=1)
        merge = layers.Flatten()(merge)

        outputs = Dense(8, activation='softmax')(merge)

        model = keras.Model(inputs, outputs)
        return model
        
    # Our model --> Optimized CNN model
    def create_CNN_test_speech_model(self, input_shape, **kwargs):
        inputs = keras.Input(shape=(input_shape))
        
        #High level features 
        high_level = layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
        high_level = layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.4)(high_level)
        
        high_level = layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.3)(high_level)
        
        high_level = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.3)(high_level)
        
        high_level = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu')(high_level)
        high_level = layers.MaxPooling1D(pool_size=3, strides = 2, padding = 'same')(high_level)
        high_level = layers.Dropout(0.3)(high_level)
        high_level = layers.Flatten()(high_level)

        outputs = layers.Dense(8, activation='softmax')(high_level)

        model = keras.Model(inputs, outputs)
        return model
    
    # Optimized LSTM model
    def create_LSTM_test_speech_model(self, input_shape, **kwargs):
        inputs = keras.Input(shape=(input_shape))

        low_level = layers.Bidirectional(LSTM(128, return_sequences=True))(inputs)
        low_level = layers.Dropout(0.5)(low_level)
        low_level = layers.Bidirectional(LSTM(64))(low_level)
        low_level = layers.Dropout(0.5)(low_level)
        low_level = layers.Dense(128)(low_level)
        low_level = layers.Dropout(0.5)(low_level)
        low_level = layers.Dense(64)(low_level)
        low_level = layers.Dropout(0.5)(low_level)
        
        outputs = layers.Dense(8, activation='softmax')(low_level)

        model = keras.Model(inputs, outputs)

        return model

    # PLotting of confusion matrix
    def plot_confusion_matrix(self, model, encoder, x_test, y_test, speech_model_type, **kwargs):
        # predicting on test data.
        pred_test = model.predict(x_test)
        y_pred = encoder.inverse_transform(pred_test)

        y_test = encoder.inverse_transform(y_test)
        df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
        df['Predicted Labels'] = y_pred.flatten()
        df['Actual Labels'] = y_test.flatten()

        df.head(10)
        cm = confusion_matrix(y_test, y_pred)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize = (12, 10))
        cmn = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
        sns.heatmap(cmn, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
        plt.title('Confusion Matrix', size=20)
        plt.xlabel('Predicted Labels', size=14)
        plt.ylabel('Actual Labels', size=14)
        plt.savefig('graphs/'+speech_model_type+'confusion.png')
        plt.show()

    def create_model(self, model_type, train_shape,  **kwargs):
        if model_type=='original_lstm':
            model = self.create_LSTM_speech_model(train_shape, **kwargs)
        elif model_type=='original_cnn':
            model = self.create_CNN_speech_model(train_shape, **kwargs)
        elif model_type=='our_model':
            model = self.create_CNN_test_speech_model(train_shape, **kwargs)
        else:
            model=self.create_original_speech_model(train_shape, **kwargs)
        return model

    
    
    