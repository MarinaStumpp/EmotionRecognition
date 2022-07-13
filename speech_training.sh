#!/bin/bash

py -3 speech_training.py \
--speech_data_available True \
--speech_model_type 'our_model' \
--speech_feature_type 'mfcc' \
--speech_epochs 10 \
--speech_runs 1
