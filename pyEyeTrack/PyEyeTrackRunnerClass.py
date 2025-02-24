import cv2
import numpy as np
import os
import threading
from datetime import datetime
import sys
from .AudioVideoRecording.VideoRecordingClass import VideoRecorder
try:
    from .AudioVideoRecording.AudioRecordingClass import AudioRecorder
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio recording will be disabled.")

from .EyeTracking.PupilTrackingClass import PupilTracking

class PyEyeTrackRunner:
    def __init__(self):
        """
        Initialize the PyEyeTrackRunner class.
        """
        self.running = True
        self.recording = False

    def check_key(self):
        """
        Check for keyboard input using cv2.waitKey instead of keyboard module.
        """
        while self.running:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                self.running = False
            elif key == ord('r'):  # Press 'r' to toggle recording
                self.recording = not self.recording

    def pyEyeTrack_runner(
        self,
        source=0,
        pupilTracking=True,
        videoRecording=True,
        audioRecording=False,
        destinationPath="./Output"
    ):
        """
        Main function to run eye tracking and recording.
        Args:
            source (int/str): Camera index or video file path (default: 0)
            pupilTracking (bool): Enable pupil tracking (default: True)
            videoRecording (bool): Enable video recording (default: True)
            audioRecording (bool): Enable audio recording (default: False)
            destinationPath (str): Output directory path (default: "./Output")
        """
        # Create output directory if it doesn't exist
        os.makedirs(destinationPath, exist_ok=True)

        # Initialize video capture
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        # Initialize components
        eyeTracking = PupilTracking(source=source) if pupilTracking else None
        videoRecorder = VideoRecorder() if videoRecording else None
        audioRecorder = None
        if audioRecording and AUDIO_AVAILABLE:
            try:
                audioRecorder = AudioRecorder()
            except Exception as e:
                print(f"Error initializing audio recorder: {e}")
                audioRecording = False

        # Start keyboard monitoring thread
        keyboard_thread = threading.Thread(target=self.check_key)
        keyboard_thread.daemon = True
        keyboard_thread.start()

        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")

        eyeTrackingOutput = {
            'timestamp': [],
            'left_pupil_x': [],
            'left_pupil_y': [],
            'right_pupil_x': [],
            'right_pupil_y': []
        }

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame for eye tracking
                if pupilTracking and eyeTracking:
                    frame, pupil_coords = eyeTracking.detect_pupil(frame)
                    if pupil_coords:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                        eyeTrackingOutput['timestamp'].append(timestamp)
                        eyeTrackingOutput['left_pupil_x'].append(pupil_coords[0][0])
                        eyeTrackingOutput['left_pupil_y'].append(pupil_coords[0][1])
                        eyeTrackingOutput['right_pupil_x'].append(pupil_coords[1][0])
                        eyeTrackingOutput['right_pupil_y'].append(pupil_coords[1][1])

                # Handle recording
                if self.recording:
                    if videoRecording and videoRecorder:
                        videoRecorder.write_frame(frame)
                    if audioRecording and audioRecorder:
                        audioRecorder.write_audio()

                # Display frame
                cv2.imshow('Eye Tracking', frame)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Cleanup
            self.running = False
            cv2.destroyAllWindows()
            cap.release()

            if videoRecording and videoRecorder:
                videoRecorder.close()
            if audioRecording and audioRecorder:
                audioRecorder.close()

            # Save eye tracking data
            if pupilTracking and eyeTracking and eyeTrackingOutput['timestamp']:
                eyeTracking.csv_writer(eyeTrackingOutput)

            print("Session ended")

                                                    
