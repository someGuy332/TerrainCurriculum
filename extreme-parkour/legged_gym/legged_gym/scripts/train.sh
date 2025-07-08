#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0  # Set GPU device

# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate MV2

# Constants
CYCLES=24
MAX_ITER=5  # Max iterations per cycle
INITIAL_GENERATION=5
EXPTID="CEP_17"  # Experiment ID
CEP_TYPE="combined_avg" # reward / goals / combined
REPLAY_TYPE="RD"
PERFORMANCE_DIR="./heightmaps/$EXPTID"  # Directory for performance history
mkdir -p "$PERFORMANCE_DIR"  # Create directory if it doesn't exist
PERFORMANCE_FILE="$PERFORMANCE_DIR/performance_history.txt"  # To store level:reward pairs

# Initialize performance file if it doesn't exist
if [ ! -f "$PERFORMANCE_FILE" ]; then
    touch "$PERFORMANCE_FILE"
fi

# Function to run training
train_in_subprocess() {
    local iteration=$1
    local selected_level=$2
    local exptid="$EXPTID" # Replace with your default or pass as arg
    local max_iter="$MAX_ITER" # Replace with your default or pass as arg
    local replay_type="$REPLAY_TYPE" # Replace with your default or pass as arg
    local resume_flag=""
    local cmd_args=""
    local log_file="training_output${exptid}_${iteration}.log"
    local cep_type="$CEP_TYPE"

    echo "Starting iteration $((iteration + 1))/$CYCLES with level: $selected_level"

    # Build command
    cmd_args="python train.py --terrain_type $selected_level --exptid $exptid --CEP --CEP_type $cep_type --no_replay --max_iterations $max_iter --headless"
    if [ "$iteration" -gt 0 ]; then
        resume_flag="--resume"
        cmd_args="$cmd_args $resume_flag"
    fi

    echo "Running: $cmd_args"
    
    # Execute command and show output in real-time while also capturing to a file
    eval "$cmd_args" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}

    output=$(<"$log_file")  # Capture the output from the log file

    if [ $exit_code -eq 0 ]; then
        echo "Iteration $((iteration + 1)) completed successfully."
        echo "$output"
    else
        echo "Iteration $((iteration + 1)) failed with exit code $exit_code"
        echo "Error output: $output"
        exit 1
    fi
}

# Main loop
main() {
    for ((i=5; i<CYCLES; i++)); do
        train_in_subprocess "$i" "$i"
    done
    # train_in_subprocess "8" "8"
}

# Run main
main
n 