#!/usr/bin/env python3
import os
import sys
import subprocess
import pkg_resources
import time
from datetime import datetime

def check_dependencies():
    """Check and install required dependencies"""
    required = {
        'numpy': 'numpy>=1.19.5',
        'pandas': 'pandas>=1.2.4',
        'opencv-python': 'opencv-python>=4.7.0',
        'keyboard': 'keyboard>=0.13.5',
        'tqdm': 'tqdm>=4.65.0',
        'setuptools': 'setuptools>=75.0.0'
    }
    
    missing = []
    
    for package, version in required.items():
        try:
            pkg_resources.require(version)
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing.append(version)
    
    if missing:
        print("Installing missing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {str(e)}")
            print("Please try installing them manually using:")
            print(f"pip install {' '.join(missing)}")
            return False
    return True

def check_model_file():
    """Check if the shape predictor model file exists and download if needed"""
    model_file = 'shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(model_file):
        print(f"{model_file} not found. Downloading automatically...")
        try:
            from pyEyeTrack.EyeTracking.AbstractEyeTrackingClass import check
            check()
            if not os.path.exists(model_file):
                print("Error: Model file download failed.")
                return False
        except Exception as e:
            print(f"Error downloading model file: {str(e)}")
            return False
    return True

def setup_output_directory():
    """Create Output directory if it doesn't exist"""
    output_dir = "Output"
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        return False

def get_session_info():
    """Get session information from user"""
    print("\n=== Eye Tracking Session Setup ===")
    print("Please enter the following information:")
    
    while True:
        participant_id = input("Participant ID: ").strip()
        if participant_id and participant_id.isalnum():
            break
        print("Participant ID must be non-empty and contain only letters and numbers.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{participant_id}_{timestamp}"
    
    return session_id

def launch_application(session_id):
    """Launch the eye tracking application"""
    try:
        from pyEyeTrack.PyEyeTrackRunnerClass import pyEyeTrack
        
        print("\nInitializing eye tracking session...")
        print(f"Session ID: {session_id}")
        print("\nInstructions:")
        print("1. Position yourself in front of the camera")
        print("2. Ensure good lighting conditions")
        print("3. Press 'ESC' to stop tracking")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        tracker = pyEyeTrack()
        tracker.pyEyeTrack_runner(
            pupilTracking=True,
            eyeTrackingLog=True,
            eyeTrackingFileName='participant',
            session_id=session_id,
            audioRecorder=False  # Disable audio recording by default
        )
        
    except ImportError as e:
        print(f"\nError importing PyEyeTrack: {str(e)}")
        print("Please ensure the package is installed correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError launching application: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the application"""
    print("=== PyEyeTrack Application ===")
    
    # Check and install dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Setup output directory
    if not setup_output_directory():
        sys.exit(1)
    
    # Check for model file
    if not check_model_file():
        sys.exit(1)
    
    # Get session information
    try:
        session_id = get_session_info()
    except KeyboardInterrupt:
        print("\nSession setup cancelled by user.")
        sys.exit(0)
    
    # Launch the application
    launch_application(session_id)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1) 