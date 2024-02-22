#!/bin/bash

# * Parameters

# Logging level for Python.
LOG_LEVEL=INFO
# Number of traces.
NUM_TRACES=4000
# Temporary collection path.
TARGET_PATH=/tmp/collect/$NUM_TRACES

# * Functions

# ** Configuration

function configure_param_json_escape_path() {
    # 1. Substitute "/" to "\/".
    # 2. Add '"' around string.
    echo \"${1//\//\\/}\"
}

function configure_param_json() {
    config_file="$1"
    param_name="$2"
    param_value="$3"
    echo "$config_file: $param_name=$param_value"
    sed -i "s/\"${param_name}\": .*,/\"${param_name}\": ${param_value},/g" "$config_file"
}

function configure_json_plot() {
    export CONFIG_JSON_PATH_SRC=$PROJECT_PATH/experiments/config/example_collection_collect_plot.json
    export CONFIG_JSON_PATH_DST=$TARGET_PATH/example_collection_collect_plot.json
    cp $CONFIG_JSON_PATH_SRC $CONFIG_JSON_PATH_DST
    configure_param_json $CONFIG_JSON_PATH_DST "trigger_threshold" "90e3"
}

function configure_json_collect() {
    export CONFIG_JSON_PATH_SRC=$PROJECT_PATH/experiments/config/example_collection_collect_plot.json
    export CONFIG_JSON_PATH_DST=$TARGET_PATH/example_collection_collect.json
    cp $CONFIG_JSON_PATH_SRC $CONFIG_JSON_PATH_DST
    configure_param_json $CONFIG_JSON_PATH_DST "trigger_threshold" "90e3"
    configure_param_json $CONFIG_JSON_PATH_DST "num_points" "$NUM_TRACES"
    configure_param_json $CONFIG_JSON_PATH_DST "fixed_key" "false"
    configure_param_json $CONFIG_JSON_PATH_DST "template_name" "$(configure_param_json_escape_path $TARGET_PATH/template.npy)"
}

# ** Instrumentation

function record() {
    plot=$1
    echo "plot=$plot"
    
    # Kill previously started radio server.
    pkill radio.py

    sudo ykushcmd -d a # power off all ykush device
    sleep 2
    sudo ykushcmd -u a # power on all ykush device
    sleep 4

    # Start SDR server.
    # NOTE: Make sure the JSON config file is configured accordingly to the SDR server here.
    $SC_SRC/radio.py --config $SC_SRC/config.toml --dir $HOME/storage/tmp --loglevel $LOG_LEVEL listen 128e6 2.512e9 8e6 --nf-id -1 --ff-id 0 --duration=0.6 --gain 76 &
    sleep 10

    # Start collection and plot result.
    sc-experiment --loglevel=$LOG_LEVEL --radio=USRP --device=$(nrfjprog --com | cut - -d " " -f 5) -o $HOME/storage/tmp/raw_0_0.npy collect $CONFIG_JSON_PATH_DST $TARGET_PATH $plot --average-out=$TARGET_PATH/template.npy
}

function analyze_only() {
    sc-experiment --loglevel=$LOG_LEVEL --radio=USRP --device=$(nrfjprog --com | cut - -d " " -f 5) -o $HOME/storage/tmp/raw_0_0.npy extract $CONFIG_JSON_PATH_DST $TARGET_PATH --plot --average-out=$TARGET_PATH/template.npy
}

# * Script

# Create collection directory.
mkdir -p $TARGET_PATH

# ** Configure the extraction / Template generation

# DONE: Set the JSON configuration file for one recording analysis.
configure_json_plot

# DONE: Use this once to record a trace. 
record --plot
# DONE: Once the recording is good, use this to configure the analysis.
# analyze_only

if [[ ! -f $TARGET_PATH/template.npy ]]; then
    echo "Template has not been created! (no file at $TARGET_PATH/template.npy)"
    exit 1
fi

# ** Collect

# DONE: Set the JSON configuration file for collection.
configure_json_collect

# DONE: Collect a set of profile traces.
record --no-plot
