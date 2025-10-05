#!/usr/bin/env python3
"""
Launcher script for VisionPivot Attendance System
This script checks dependencies and launches the main application.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_modules = [
        'cv2', 'torch', 'facenet_pytorch', 'firebase_admin', 
        'pyttsx3', 'PyQt5', 'yagmail', 'pandas', 'numpy', 'PIL'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'torch':
                import torch
            elif module == 'facenet_pytorch':
                from facenet_pytorch import InceptionResnetV1, MTCNN
            elif module == 'firebase_admin':
                import firebase_admin
            elif module == 'pyttsx3':
                import pyttsx3
            elif module == 'PyQt5':
                from PyQt5.QtWidgets import QApplication
            elif module == 'yagmail':
                import yagmail
            elif module == 'pandas':
                import pandas
            elif module == 'numpy':
                import numpy
            elif module == 'PIL':
                from PIL import Image
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['Images', 'Encodings', 'Attendance']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def main():
    """Main launcher function."""
    print("VisionPivot Attendance System Launcher")
    print("=" * 40)
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please run 'pip install -r requirements.txt' to install missing packages.")
        print("Or run 'install_dependencies.bat' on Windows.")
        input("Press Enter to exit...")
        return False
    
    print("All dependencies found!")
    
    # Create directories
    print("Creating directories...")
    create_directories()
    
    # Check for Firebase config
    if not os.path.exists("vision-pivot-firebase-adminsdk-7j9hj-b5fe1eafa5.json"):
        print("Warning: Firebase configuration file not found!")
        print("Please ensure you have the correct Firebase service account key file.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Launch the application
    print("Launching VisionPivot Attendance System...")
    try:
        # Import the main application module
        import visionpivot_final
        # The application should start automatically when imported due to the __main__ block
        print("Application launched successfully!")
    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        return False
    
    return True

if __name__ == "__main__":
    main()
