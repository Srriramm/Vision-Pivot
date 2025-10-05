#!/usr/bin/env python3
"""
Setup script for VisionPivot Attendance System
This script helps set up the environment and dependencies for the application.
"""

import os
import sys
import subprocess
import shutil

def create_directories():
    """Create necessary directories for the application."""
    directories = ['Images', 'Encodings', 'Attendance']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed all requirements!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def check_firebase_config():
    """Check if Firebase configuration file exists."""
    firebase_config = "vision-pivot-firebase-adminsdk-7j9hj-b5fe1eafa5.json"
    if os.path.exists(firebase_config):
        print(f"Firebase configuration file found: {firebase_config}")
        return True
    else:
        print(f"Warning: Firebase configuration file not found: {firebase_config}")
        print("Please ensure you have the correct Firebase service account key file.")
        return False

def check_background_image():
    """Check if background image exists."""
    background_image = "Lawrencium.jpg"
    if os.path.exists(background_image):
        print(f"Background image found: {background_image}")
        return True
    else:
        print(f"Warning: Background image not found: {background_image}")
        print("The application will use a default background color instead.")
        return False

def main():
    """Main setup function."""
    print("Setting up VisionPivot Attendance System...")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return False
    
    # Check configuration files
    print("\n3. Checking configuration files...")
    check_firebase_config()
    check_background_image()
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Add employee images to the 'Images' folder (create subfolders for each employee)")
    print("2. Run 'python face_embeddings.py' to generate face encodings")
    print("3. Run 'python visionpivot_final.py' to start the application")
    print("\nNote: Make sure you have the correct Firebase configuration file.")
    
    return True

if __name__ == "__main__":
    main()
