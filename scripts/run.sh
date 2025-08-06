#!/bin/bash

TYPE=${1:-train} # train or play, default is 'train'
RUN_NAME=${2:-""} # run name when training and checkpoint name when playing

# Shift only the arguments that were actually provided (up to 2)
if [ $# -ge 2 ]; then
    shift 2
elif [ $# -eq 1 ]; then
    shift 1
fi

RL_LIB="rsl_rl"
RL_ENV_NAME="Gen3-Skimmer-v0"
TRAIN_SCRIPT="train.py"
PLAY_SCRIPT="play_skimmer.py"
N_ITER=500
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_FULL_NAME="/workspace/kinova_isaaclab_sim2real/logs/$RL_LIB/skimmer_gen3/$RUN_NAME"
PYTHON="/isaac-sim/python.sh"

if [ "$TYPE" == "play" ]; then
    RUN_FULL_NAME=$BASE_FULL_NAME/model_$((N_ITER-1)).pt
    echo "RUN_FULL_NAME (play): $RUN_FULL_NAME"
elif [ "$TYPE" == "train" ]; then
    echo "RUN_FULL_NAME (train): $BASE_FULL_NAME"
fi

# Check if the RL_LIB directory exists
if [ ! -d "$SCRIPT_DIR/$RL_LIB" ]; then
    echo "Error: RL_LIB directory not found: $SCRIPT_DIR/$RL_LIB"
    exit 1
fi

if [ -z "$RUN_NAME" ] || [[ ! "$RUN_NAME" =~ ^[0-9]{2,} ]]; then
    echo "Error: Second argument (RUN_NAME) must exist and start with at least two digits."
    exit 1
fi

if [ "$TYPE" == "train" ]; then
    CMD="$PYTHON $SCRIPT_DIR/$RL_LIB/$TRAIN_SCRIPT --task $RL_ENV_NAME --max_iterations $N_ITER --run_name $RUN_NAME $*"
    echo "$CMD"
    eval $CMD
elif [ "$TYPE" == "play" ]; then
    CMD="$PYTHON $SCRIPT_DIR/$RL_LIB/$PLAY_SCRIPT --task $RL_ENV_NAME --checkpoint $RUN_FULL_NAME $*"
    echo "$CMD"
    eval $CMD
else
    echo "Invalid type: $TYPE"
    exit 1
fi
