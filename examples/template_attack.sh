#!/bin/bash

# * Parameters

# Path of dataset used to create the profile.
TRAIN_SET=/tmp/collect/4000
# Path used to store the created profile.
PROFILE_PATH=$TRAIN_SET/profile
# Path of dataset used to perform the attack.
ATTACK_SET=/tmp/collect/500

# Number of traces to use.
NUM_TRACES=3000
# Delimiters.
START_POINT=0
END_POINT=0

# * Functions

function profile() {
    echo "Press 's' to save figs to ~/Figure_1.png and ~/Figure_2.png"
    sc-attack --plot --norm --data-path $TRAIN_SET --start-point $START_POINT --end-point $END_POINT --num-traces $NUM_TRACES profile $PROFILE_PATH --pois-algo r --num-pois 1 --poi-spacing 2 --variable p_xor_k
    mv ~/Figure_1.png $PROFILE_PATH/plot_mean_trace.png
    mv ~/Figure_2.png $PROFILE_PATH/plot_poi_1.png
}

function attack() {
    sc-attack --plot --norm --data-path $ATTACK_SET --start-point $START_POINT --end-point $END_POINT --num-traces $NUM_TRACES --bruteforce attack $PROFILE_PATH --attack-algo pcc --variable p_xor_k
}

# * Script

# Initialize directories.
mkdir -p $PROFILE_PATH

# Create a profile.
profile

if [[ ! -f $PROFILE_PATH/PROFILE_MEAN_TRACE.npy ]]; then
    echo "Profile has not been created! (no file at $PROFILE_PATH/*.npy)"
    exit 1
fi

# Attack using previously created template.
attack
