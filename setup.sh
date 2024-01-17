#!/bin/bash

# Check if ROS environment variables are set
if [ -n "$ROS_DISTRO" ] && [ -n "$ROS_PACKAGE_PATH" ]; then
    echo "ROS is installed."
    echo "ROS distribution: $ROS_DISTRO"
    echo "ROS package path: $ROS_PACKAGE_PATH"
else
    echo "ROS is not installed or environment variables are not set."
    exit
fi


pip install "cython<3"
pip install -e .