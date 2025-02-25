from pyEyeTrack import PyEyeTrackRunner
import sys
import cv2
import json
from datetime import datetime
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pyEyeTrack.EyeTracking.PupilTrackingClass import PupilTracking
import time
import tempfile
import shutil
try:
    from pyEyeTrack.AudioVideoRecording.AudioRecordingClass import AudioRecorder
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

def get_session_metadata():
    """Collect detailed session metadata"""
    # First determine if this is a pilot or live session
    while True:
        session_type = input("\nIs this a pilot test session? (yes/no): ").strip().lower()
        if session_type in ['yes', 'no']:
            break
        print("Please enter 'yes' or 'no'")
    
    metadata = {
        'session_type': 'pilot' if session_type == 'yes' else 'live',
        'participant': {
            'id': input("Please enter participant ID: ").strip(),
            'age': input("Participant age (optional, press Enter to skip): ").strip() or None,
            'gender': input("Participant gender (optional, press Enter to skip): ").strip() or None,
            'vision_correction': input("Vision correction? (glasses/contacts/none): ").strip().lower() or 'none'
        },
        'session': {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'lighting_condition': input("Lighting condition (bright/dim/dark): ").strip().lower() or 'bright',
            'distance_from_screen': input("Approximate distance from screen in cm: ").strip() or '60',
            'screen_resolution': input("Screen resolution (e.g., 1920x1080): ").strip() or '1920x1080'
        },
        'experiment': {
            'task_type': input("Task type (free-viewing/reading/tracking): ").strip().lower() or 'free-viewing',
            'duration': input("Planned duration in minutes: ").strip() or '5',
            'notes': input("Additional notes (optional): ").strip() or None
        }
    }
    
    return metadata

def validate_metadata(metadata):
    """Validate required metadata fields"""
    if not metadata['participant']['id']:
        raise ValueError("Participant ID cannot be empty")
    
    # Convert numeric fields
    try:
        if metadata['participant']['age']:
            metadata['participant']['age'] = int(metadata['participant']['age'])
        metadata['session']['distance_from_screen'] = float(metadata['session']['distance_from_screen'])
        metadata['experiment']['duration'] = float(metadata['experiment']['duration'])
    except ValueError as e:
        raise ValueError(f"Invalid numeric value in metadata: {str(e)}")

def setup_session_directory(metadata):
    """Create and setup session directory structure"""
    base_dir = "Output"
    
    # Create pilot or live data subdirectory
    data_type_dir = "pilot_data" if metadata['session_type'] == 'pilot' else "live_data"
    
    session_dir = os.path.join(
        base_dir,
        data_type_dir,
        f"participant_{metadata['participant']['id']}",
        f"session_{metadata['session']['id']}"
    )
    
    # Create directory structure with more detailed organization
    subdirs = [
        'data/raw_data',          # Raw eye tracking data
        'data/processed_data',    # Processed/analyzed data
        'metadata',               # Session metadata
        'analysis',               # Analysis results
        'logs',                   # Session logs
        'exports'                 # Exported visualizations/reports
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(session_dir, subdir), exist_ok=True)
    
    # Save metadata
    metadata_file = os.path.join(session_dir, 'metadata', 'session_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Create a session log file
    log_file = os.path.join(session_dir, 'logs', 'session.log')
    with open(log_file, 'w') as f:
        f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session type: {metadata['session_type']}\n")
        f.write(f"Participant ID: {metadata['participant']['id']}\n")
        f.write(f"Session ID: {metadata['session']['id']}\n")
    
    return session_dir

def main():
    try:
        # Collect and validate session metadata
        metadata = get_session_metadata()
        validate_metadata(metadata)
        
        # Setup session directory
        session_dir = setup_session_directory(metadata)
        
        print("\nSession Information:")
        print(f"Participant ID: {metadata['participant']['id']}")
        print(f"Session ID: {metadata['session']['id']}")
        print(f"Task Type: {metadata['experiment']['task_type']}")
        print(f"Duration: {metadata['experiment']['duration']} minutes")
        
        # Initialize the eye tracker with metadata
        tracker = PyEyeTrackRunner(
            participant_id=metadata['participant']['id'],
            session_id=metadata['session']['id'],
            metadata=metadata
        )
        
        print("\nStarting eye tracking...")
        print("Controls:")
        print("Press 'r' to start/stop recording")
        print("Press 'q' to quit")
        print("Press 's' to add a marker/note")
        print("Press 'p' to pause/resume\n")
        
        # Run the eye tracking
        tracker.pyEyeTrack_runner(
            source=0,  # Use default webcam
            pupilTracking=True,  # Enable pupil tracking
            videoRecording=False,  # Disable video recording
            audioRecording=False,  # Disable audio recording
            destinationPath=session_dir  # Use organized session directory
        )
        
    except KeyboardInterrupt:
        print("\nEye tracking stopped by user.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()

class TestPyEyeTrackBase(unittest.TestCase):
    """Base test class with common setup and teardown"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_filename = "test_eye_tracking_data.csv"
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests"""
        shutil.rmtree(cls.test_dir)

class TestPyEyeTrackDataHandling(TestPyEyeTrackBase):
    """Test data handling and validation"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session"
        )
        self.tracker.data_dir = self.test_dir
        
        # Initialize eye_data with consistent lengths
        self.tracker.eye_data = {
            'timestamp': [],
            'frame_number': [],
            'left_eye_x': [],
            'left_eye_y': [],
            'right_eye_x': [],
            'right_eye_y': [],
            'left_pupil_size': [],
            'right_pupil_size': [],
            'left_blink': [],
            'right_blink': [],
            'gaze_x': [],
            'gaze_y': [],
            'head_pose_x': [],
            'head_pose_y': [],
            'head_pose_z': [],
            'marker': [],
            'data_quality': []
        }

    def test_validate_eye_data(self):
        """Test eye data validation"""
        # Test valid data
        valid_data = {
            'left_pupil_size': 5.0,
            'gaze_x': 960,
            'gaze_y': 540,
            'head_pose_x': 45
        }
        quality_score = self.tracker.validate_eye_data(valid_data)
        self.assertEqual(quality_score, 1.0)
        
        # Test invalid data
        invalid_data = {
            'left_pupil_size': 10.0,  # Out of range
            'gaze_x': 2000,  # Out of range
            'gaze_y': 540
        }
        quality_score = self.tracker.validate_eye_data(invalid_data)
        self.assertLess(quality_score, 1.0)
        
        # Test None data
        quality_score = self.tracker.validate_eye_data(None)
        self.assertEqual(quality_score, 0.0)
        
        # Test invalid type data
        quality_score = self.tracker.validate_eye_data("invalid")
        self.assertEqual(quality_score, 0.0)

    def test_save_eye_data(self):
        """Test saving eye tracking data"""
        # Add test data with consistent lengths
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        num_samples = 5
        
        for i in range(num_samples):
            self.tracker.eye_data['timestamp'].append(timestamp)
            self.tracker.eye_data['frame_number'].append(i)
            self.tracker.eye_data['left_eye_x'].append(100 + i)
            self.tracker.eye_data['left_eye_y'].append(150 + i)
            self.tracker.eye_data['right_eye_x'].append(200 + i)
            self.tracker.eye_data['right_eye_y'].append(250 + i)
            self.tracker.eye_data['left_pupil_size'].append(5.0)
            self.tracker.eye_data['right_pupil_size'].append(5.0)
            self.tracker.eye_data['left_blink'].append(0)
            self.tracker.eye_data['right_blink'].append(0)
            self.tracker.eye_data['gaze_x'].append(350 + i)
            self.tracker.eye_data['gaze_y'].append(400 + i)
            self.tracker.eye_data['head_pose_x'].append(10)
            self.tracker.eye_data['head_pose_y'].append(20)
            self.tracker.eye_data['head_pose_z'].append(30)
            self.tracker.eye_data['marker'].append(None)
            self.tracker.eye_data['data_quality'].append(1.0)
        
        # Set start time for recording stats
        self.tracker.start_time = datetime.now().timestamp()
        
        # Save data
        self.tracker.save_eye_data()
        
        # Verify files were created
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.endswith('.csv') for f in files))
        
        # Verify data was cleared after saving
        self.assertEqual(len(self.tracker.eye_data['timestamp']), 0)
        
        # Load and verify saved data
        csv_file = next(f for f in files if f.endswith('.csv'))
        df = pd.read_csv(os.path.join(self.test_dir, csv_file))
        self.assertEqual(len(df), num_samples)
        self.assertTrue(all(col in df.columns for col in self.tracker.eye_data.keys()))

class TestPyEyeTrackRecording(TestPyEyeTrackBase):
    """Test recording functionality"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session"
        )
        self.tracker.data_dir = self.test_dir
        self.tracker.start_time = datetime.now().timestamp()

    def test_recording_controls(self):
        """Test recording controls"""
        # Start recording
        self.tracker.start_recording()
        self.assertTrue(self.tracker.recording)
        
        # Add some test data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.tracker.eye_data['timestamp'].append(timestamp)
        self.tracker.eye_data['frame_number'].append(1)
        self.tracker.eye_data['left_eye_x'].append(100)
        self.tracker.eye_data['left_eye_y'].append(150)
        self.tracker.eye_data['right_eye_x'].append(200)
        self.tracker.eye_data['right_eye_y'].append(250)
        self.tracker.eye_data['left_pupil_size'].append(5.0)
        self.tracker.eye_data['right_pupil_size'].append(5.0)
        self.tracker.eye_data['left_blink'].append(0)
        self.tracker.eye_data['right_blink'].append(0)
        self.tracker.eye_data['gaze_x'].append(350)
        self.tracker.eye_data['gaze_y'].append(400)
        self.tracker.eye_data['head_pose_x'].append(10)
        self.tracker.eye_data['head_pose_y'].append(20)
        self.tracker.eye_data['head_pose_z'].append(30)
        self.tracker.eye_data['marker'].append(None)
        self.tracker.eye_data['data_quality'].append(1.0)
        
        # Stop recording
        self.tracker.stop_recording()
        self.assertFalse(self.tracker.recording)
        
        # Verify data was saved
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.endswith('.csv') for f in files))

    def test_add_marker(self):
        """Test adding markers during recording"""
        marker_text = "Test marker"
        self.tracker.frame_count = 0
        self.tracker.add_marker(marker_text)
        
        self.assertEqual(len(self.tracker.markers), 1)
        self.assertEqual(self.tracker.markers[0]['note'], marker_text)
        self.assertEqual(self.tracker.markers[0]['frame_number'], 0)

    def test_add_note(self):
        """Test adding notes"""
        note_text = "Test note"
        self.tracker.add_note(note_text)
        
        self.assertEqual(len(self.tracker.notes), 1)
        self.assertEqual(self.tracker.notes[0]['note'], note_text)

@unittest.skipIf(not AUDIO_AVAILABLE, "PyAudio not available")
class TestPyEyeTrackAudioVideo(TestPyEyeTrackBase):
    """Test audio and video recording"""
    
    def setUp(self):
        """Set up each test"""
        self.video_recorder = AudioRecorder(file_name="test_video")
        if AUDIO_AVAILABLE:
            self.audio_recorder = AudioRecorder(file_name="test_audio")

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'video_recorder'):
            self.video_recorder.stop()
        if AUDIO_AVAILABLE and hasattr(self, 'audio_recorder'):
            self.audio_recorder.stop()
        
        # Clean up files
        for file in ['test_video.avi', 'test_audio.wav']:
            if os.path.exists(file):
                os.remove(file)

    def test_video_recording(self):
        """Test video recording"""
        self.video_recorder.main()
        time.sleep(1)  # Record for 1 second
        self.video_recorder.stop()
        
        self.assertTrue(os.path.exists('test_video.avi'))
        self.assertGreater(os.path.getsize('test_video.avi'), 0)

    @unittest.skipIf(not AUDIO_AVAILABLE, "PyAudio not available")
    def test_audio_recording(self):
        """Test audio recording"""
        self.audio_recorder.main()
        time.sleep(1)  # Record for 1 second
        self.audio_recorder.stop()
        
        self.assertTrue(os.path.exists('test_audio.wav'))
        self.assertGreater(os.path.getsize('test_audio.wav'), 0)

def main():
    """Run the test suite"""
    unittest.main()

if __name__ == '__main__':
    main() 