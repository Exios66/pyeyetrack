from pyEyeTrack import PyEyeTrackRunner

def main():
    # Initialize the eye tracker
    tracker = PyEyeTrackRunner()
    
    # Run the eye tracking
    tracker.pyEyeTrack_runner(
        source=0,  # Use default webcam
        pupilTracking=True,  # Enable pupil tracking
        videoRecording=True,  # Enable video recording
        audioRecording=False,  # Disable audio recording since PyAudio is not available
        destinationPath="Output"  # Output directory
    )

if __name__ == "__main__":
    main() 