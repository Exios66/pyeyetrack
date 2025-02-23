from pyEyeTrack.PyEyeTrackRunnerClass import pyEyeTrack

def main():
    # Initialize the eye tracker
    tracker = pyEyeTrack()
    
    # Run pupil tracking with default webcam
    # Save tracking data to 'eye_tracking_test.csv' in the Output directory
    tracker.pyEyeTrack_runner(
        pupilTracking=True,  # Enable pupil tracking
        blinkDetection=False,  # Disable blink detection for now
        video_source=0,  # Use default webcam
        eyeTrackingLog=True,  # Save tracking data
        eyeTrackingFileName='eye_tracking_test',  # Output filename
        destinationPath='./Output'  # Output directory
    )

if __name__ == "__main__":
    main() 