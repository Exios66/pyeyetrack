from pyEyeTrack.EyeTracking.PupilTrackingClass import PupilTracking
from pyEyeTrack.EyeTracking.BlinkingClass import Blinking
from pyEyeTrack.EyeTracking.PupilBlinkingClass import PupilBlinking
from pyEyeTrack.AudioVideoRecording.VideoRecordingClass import VideoRecorder
import threading
import importlib
import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Try to import AudioRecorder, but don't fail if PyAudio is not available
try:
    from pyEyeTrack.AudioVideoRecording.AudioRecordingClass import AudioRecorder
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio recording will be disabled.")

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

    def dynamic_import(self, module):
        return importlib.import_module(module)

    def pyEyeTrack_runner(
            self,
            UI=False,
            UI_file_name="User_ImageUI_EscExit",
            pupilTracking=False,
            blinkDetection=False,
            video_source=0,
            eyeTrackingLog=True,
            eyeTrackingFileName='EyeTrackLog',
            videoRecorder=False,
            videoName='video',
            audioRecorder=False,
            audioName='audio',
            destinationPath='/Output',
            session_id=None):
        """
        This function enables the user to run the functionalities of the 
        library simultaneously.
        Functionalities include running the UI specified by the user, 
        pupil tracking, blink detection, video recording and audio recording.
        The user can set flags to run the combination of these functionalities. 
        The function also allows the user to name the output file.

        Args:
            UI (bool, optional): This parameter enables the user to run UI. 
            Default: False.

            UI_file_name (str, optional): This parameter takes the file name 
            of the UI. Default: "User_ImageUI_EscExit".

            pupilTracking (bool, optional): This parameter enables the user to 
            run pupil tracking. Default: False.

            blinkDetection (bool, optional): This parameter enables the user 
            to run blink detection. Default: False.

            video_source (int/str, optional): This parameter takes either 
            device index or a video file as input. Default: 0.

            eyeTrackingLog (bool, optional): This parameter enables the user to 
            generate a CSV of pupil tracking/ blink detection. Default: True.

            eyeTrackingFileName (str, optional): This parameter takes the file name 
            for the CSV. Default: 'EyeTrackLog'.

            videoRecorder (bool, optional): This parameter enables the user to 
            record video. Default: False.

            videoName (str, optional): This parameter enables the user to specify 
            the filename with which the recorded video is to be saved.
            Default: 'video'.

            audioRecorder (bool, optional): This parameter enables the user to 
            record audio. Default: False.

            audioName (str, optional):  This parameter enables the user to specify 
            the filename with which the recorded video is to be saved.
            Default: 'audio'.

            destinationPath (str, optional): The parameter enables the user to specify 
            the location of the output files. Default: '/Output'.

            session_id (str, optional): Unique identifier for the tracking session.
            If not provided, a timestamp-based ID will be generated. Default: None.
        """

        if audioRecorder and not AUDIO_AVAILABLE:
            print("Warning: Audio recording requested but PyAudio is not available. Audio recording will be skipped.")
            audioRecorder = False

        startEyeTracking = False
        outputPath = destinationPath

        if os.access(
                destinationPath,
                os.W_OK) == False and destinationPath != '/Output':
            print('You may not have write permission.Try changing the destination path.')
            sys.exit()

        if os.path.exists(
                destinationPath) == False and destinationPath != '/Output':
            os.mkdir(destinationPath)
        elif destinationPath == '/Output':
            currentPath = os.getcwd()
            outputPath = currentPath + '/Output'
            if os.path.exists(outputPath) == False:
                os.mkdir(outputPath)

        outputPath = outputPath + '/'

        if (pupilTracking or blinkDetection) and videoRecorder:
            print('Video Recording and Eye Tracking functionalities ' 
            'require access to the webcam simultaneously and are therefore ' 
            'recommended not to run these functionalities simultaneously.')
            sys.exit()

       
        if pupilTracking or blinkDetection:
            startEyeTracking = True

        if video_source != 0:
            if os.path.exists(video_source) == False:
                print('Please specify correct path for the video source.')
                sys.exit()

        if blinkDetection and pupilTracking:
            eyeTracking = PupilBlinking(video_source)
            eyeTrackingThread = threading.Thread(target=eyeTracking.start)

        if blinkDetection and pupilTracking == False:
            eyeTracking = Blinking(video_source)
            eyeTrackingThread = threading.Thread(target=eyeTracking.start)

        if pupilTracking and blinkDetection == False:
            eyeTracking = PupilTracking(video_source, session_id)
            eyeTrackingThread = threading.Thread(target=eyeTracking.start)

        if videoRecorder:
            videoOutputPath = outputPath + videoName
            videoRecorder = VideoRecorder(videoOutputPath)
            videoRecorderThread = threading.Thread(target=videoRecorder.main)

        if audioRecorder:
            try:
                audioOutputPath = outputPath + audioName
                audioRecorder = AudioRecorder(outputPath + audioName)
                audioRecorderThread = threading.Thread(target=audioRecorder.main)
            except Exception as e:
                print(f"Warning: Failed to initialize audio recording: {str(e)}")
                audioRecorder = False

        if UI:
            module = self.dynamic_import(UI_file_name)
            if hasattr(module, 'main'):
                uiThread = threading.Thread(target=module.main)
            else:
                print(
                    'UI needs a main method. Please Refer documentation for more information.')
                sys.exit()

        if UI:
            uiThread.start()

        if startEyeTracking:
            eyeTrackingThread.start()
            
        if videoRecorder:
            videoRecorderThread.start()

        if audioRecorder:
            audioRecorderThread.start()

        if UI:
            uiThread.join()

        if startEyeTracking:
            eyeTrackingThread.join()
            if eyeTrackingLog:
                eyeTrackingOutput = outputPath + eyeTrackingFileName
                eyeTracking.csv_writer(eyeTrackingOutput)

        if videoRecorder:
            videoRecorderThread.join()
            videoRecorder.stop()

        if audioRecorder:
            audioRecorderThread.join()
            audioRecorder.stop()

                                                    
