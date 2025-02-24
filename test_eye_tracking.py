from pyEyeTrack import PyEyeTrackRunner
import cv2
import sys
import time

def main():
    try:
        # Initialize the eye tracker
        tracker = PyEyeTrackRunner()
        
        print("Starting eye tracking...")
        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")
        
        # Run the eye tracking
        tracker.pyEyeTrack_runner(
            source=0,  # Use default webcam
            pupilTracking=True,  # Enable pupil tracking
            videoRecording=True,  # Enable video recording
            audioRecording=False,  # Disable audio recording
            destinationPath="Output"  # Output directory
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 