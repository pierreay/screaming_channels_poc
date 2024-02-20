#!/bin/bash

function record_and_analyze() {
    # Kill previously started radio server.
    pkill radio.py

    sudo ykushcmd -d a # power off all ykush device
    sleep 2
    sudo ykushcmd -u a # power on all ykush device
    sleep 4

    # Start SDR server.
    # NOTE: Make sure the JSON config file is configured accordingly to the SDR server here.
    $SC_SRC/radio.py --config $SC_SRC/config.toml --dir $HOME/storage/tmp --loglevel DEBUG listen 128e6 2.512e9 8e6 --nf-id -1 --ff-id 0 --duration=1 --gain 76 &
    sleep 10

    # Start collection and plot result.
    sc-experiment --loglevel=DEBUG --radio=USRP --device=$(nrfjprog --com | cut - -d " " -f 5) -o $HOME/storage/tmp/raw_0_0.npy collect $PROJECT_PATH/experiments/config/example_collection_collect_plot.json /tmp/collect --plot
}

function analyze_only() {
    sc-experiment --loglevel=DEBUG --radio=USRP --device=$(nrfjprog --com | cut - -d " " -f 5) -o $HOME/storage/tmp/raw_0_0.npy extract $PROJECT_PATH/experiments/config/example_collection_collect_plot.json /tmp/collect --plot
}

# Use this once to record a trace. 
# record_and_analyze
# Once the recording is good, use this to configure the analysis.
# analyze_only
