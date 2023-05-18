#!/bin/bash

# Define variables
REPO_NAME="python-dev-env"
PYTHON_VERSION="3.11.3"

# Create a new directory for the repository
mkdir $REPO_NAME
cd $REPO_NAME

# Initialize a new Git repository
git init

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install pylint black

# Create a requirements.txt file
pip freeze > requirements.txt

# Create a main.py file (optional)
echo 'print("Hello, world!")' > main.py

# Commit the initial files
git add .
git commit -m "Initial commit"

# Display completion message
echo "Python development environment repository created successfully."
echo "To activate the virtual environment, run: source venv/bin/activate"
