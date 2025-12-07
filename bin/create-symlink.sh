#!/bin/bash

echo "Checking if virtual environment already exists"
if [ -d "/data/msc-proj/g5env" ]; then
    echo "Virtual environment already exists"
    exit 0
fi

echo "Creating virtual environment in /data/msc-proj/g5env"
uv venv --python 3.12 /data/msc-proj/g5env
echo "Created virtual environment in /data/msc-proj/g5env"

ln -s /data/msc-proj/g5env .venv
echo "Created symlink in .venv"

# For verifying the symlink
echo "Verifying the symlink"
echo "$(ls -l .venv)" 