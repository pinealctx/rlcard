#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./upload2host.sh <username@hostname:/your_destination_dir>"
    exit 1
fi

# Enable command echo
set -x

# Assign argument to a variable
DESTINATION=$1

# Create the destination directory on the remote host
ssh "${DESTINATION%:*}" mkdir -p "${DESTINATION#*:}"

# Define directories to copy
DIRECTORIES=("examples" "new_agents" "our_agent" "rlcard" "agent_test")

# Loop over directories and copy each one
for dir in "${DIRECTORIES[@]}"; do
    scp -r "$dir" "$DESTINATION"
done

# Disable command echo
set +x
