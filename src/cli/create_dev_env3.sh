#!/bin/bash

# Define variables
REPO_NAME="python-dev-env"

# Create a new directory for the repository
mkdir $REPO_NAME
cd $REPO_NAME

# Create subdirectories
mkdir src
mkdir tests
mkdir docs
mkdir data

# Create README.md file
touch README.md

# Create .gitignore file
echo "venv/" > .gitignore

# Display completion message
echo "Folder structure for Python development environment created successfully."
