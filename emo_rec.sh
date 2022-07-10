#!/bin/bash

py -3 speech_prediction.py \
--speech_data_available False \
--speech_model_type 'our_model' \
--speech_feature_type 'mfcc' \
--speech_epochs 100 \
--speech_runs 1
