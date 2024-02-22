#!/bin/bash

# * Parameters

# Path for collection.
TARGET_PATH=/tmp/collect/4000
# Path for profile.
PROFILE_PATH=$TARGET_PATH/profile

# Number of traces to use.
NUM_TRACES=2500
# Delimiters.
START_POINT=0
END_POINT=0

# * Functions

function profile() {
    echo "Press 's' to save figs to ~/Figure_1.png and ~/Figure_2.png"
    sc-attack --plot --norm --data-path $TARGET_PATH --start-point $START_POINT --end-point $END_POINT --num-traces $NUM_TRACES profile $PROFILE_PATH --pois-algo r --num-pois 1 --poi-spacing 2 --variable p_xor_k
    mv ~/Figure_1.png $PROFILE_PATH/profile_mean_trace.png
    mv ~/Figure_2.png $PROFILE_PATH/profile_poi_1.png
}

function attack() {
    sc-attack --plot --norm --data-path $TARGEt_PATH --start-point $START_POINT --end-point $END_POINT --num-traces $NUM_TRACES --bruteforce attack $PROFILE_PATH --attack-algo pcc --variable p_xor_k
}

# * Script

# Initialize directories.
mkdir -p $PROFILE_PATH

# DONE: Create a profile.
profile

if [[ ! -f $PROFILE_PATH/PROFILE_MEAN_TRACE.npy ]]; then
    echo "Profile has not been created! (no file at $PROFILE_PATH/*.npy)"
    exit 1
fi

# TODO: Attack using previously created template.
# attack
