#!/usr/bin/env python3
"""
Simple launcher for VisionPivot Attendance System
"""

import sys
import os

def main():
    """Launch the VisionPivot application."""
    print("Starting VisionPivot Attendance System...")
    
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create necessary directories
    directories = ['Images', 'Encodings', 'Attendance']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    try:
        # Import and run the main application
        from visionpivot_final import AttendanceApp
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        window = AttendanceApp()
        window.show()
        
        print("Application started successfully!")
        print("Close the application window to exit.")
        
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
