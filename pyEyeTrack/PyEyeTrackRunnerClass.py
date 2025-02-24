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
        self.window_name = "PyEyeTrack"
        cv2.namedWindow(self.window_name)

    def check_key(self):
        """
        Check for keyboard input using cv2.waitKey.
        """
        try:
            while self.running:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    self.recording = not self.recording
                    status = "Started" if self.recording else "Stopped"
                    print(f"Recording {status}")
        except Exception as e:
            print(f"Error in key checking: {str(e)}")
            self.running = False

    def cleanup(self):
        """
        Clean up resources.
        """
        cv2.destroyAllWindows()
        self.running = False

    def pyEyeTrack_runner(
        self,
        source=0,
        pupilTracking=True,
        videoRecording=True,
        audioRecording=False,
        destinationPath="./Output"
    ):
        """
        Main runner method for eye tracking.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(destinationPath, exist_ok=True)

            # Initialize video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise Exception("Could not open video source")

            # Start key checking thread
            key_thread = threading.Thread(target=self.check_key)
            key_thread.daemon = True
            key_thread.start()

            # Initialize components
            if pupilTracking:
                pupil_tracker = PupilTracking()

            if videoRecording:
                video_recorder = VideoRecorder(destinationPath)

            if audioRecording and AUDIO_AVAILABLE:
                audio_recorder = AudioRecorder(destinationPath)

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                if pupilTracking:
                    frame = pupil_tracker.pupil_tracking(frame)

                if self.recording:
                    if videoRecording:
                        video_recorder.write_frame(frame)
                    if audioRecording and AUDIO_AVAILABLE:
                        audio_recorder.write_audio()

                # Show the frame
                cv2.imshow(self.window_name, frame)

        except Exception as e:
            print(f"Error in eye tracking: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            self.cleanup()

                                                    
