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
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_metadata = {
            'participant': {'id': 'test_participant'},
            'session': {'id': 'test_session'},
            'experiment': {
                'task_type': 'test_task',
                'duration': 5
            }
        }
        
        # Create test image
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(cls.test_image, (320, 240), 5, (255, 255, 255), -1)  # Add white dot
        
        # Mock camera
        cls.mock_cap = Mock()
        cls.mock_cap.read.return_value = (True, cls.test_image)
        cls.mock_cap.isOpened.return_value = True

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)

class TestPyEyeTrackInitialization(TestPyEyeTrackBase):
    """Test initialization and basic setup"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session",
            metadata=self.test_metadata
        )

    def test_initialization(self):
        """Test proper initialization of PyEyeTrack"""
        self.assertEqual(self.tracker.participant_id, "test_participant")
        self.assertEqual(self.tracker.session_id, "test_session")
        self.assertEqual(self.tracker.metadata, self.test_metadata)
        self.assertTrue(self.tracker.running)
        self.assertFalse(self.tracker.recording)
        self.assertFalse(self.tracker.paused)
        
        # Test data structure initialization
        self.assertIn('timestamp', self.tracker.eye_data)
        self.assertIn('frame_number', self.tracker.eye_data)
        self.assertIn('left_eye_x', self.tracker.eye_data)
        self.assertIn('data_quality', self.tracker.eye_data)

    def test_validation_ranges(self):
        """Test validation ranges are properly set"""
        self.assertIn('pupil_size', self.tracker.validation_ranges)
        self.assertIn('gaze_x', self.tracker.validation_ranges)
        self.assertIn('gaze_y', self.tracker.validation_ranges)
        self.assertIn('head_pose', self.tracker.validation_ranges)
        
        # Test range values
        self.assertEqual(self.tracker.validation_ranges['gaze_x'], (0, 1920))
        self.assertEqual(self.tracker.validation_ranges['gaze_y'], (0, 1080))

class TestPyEyeTrackDataHandling(TestPyEyeTrackBase):
    """Test data handling and validation"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session"
        )
        self.tracker.data_dir = self.test_dir

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
        with self.assertRaises(TypeError):
            self.tracker.validate_eye_data(None)

    def test_save_eye_data(self):
        """Test saving eye tracking data"""
        # Add test data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.tracker.eye_data['timestamp'].append(timestamp)
        self.tracker.eye_data['frame_number'].append(1)
        self.tracker.eye_data['left_eye_x'].append(100)
        self.tracker.eye_data['left_eye_y'].append(100)
        self.tracker.eye_data['right_eye_x'].append(200)
        self.tracker.eye_data['right_eye_y'].append(200)
        
        # Save data
        self.tracker.save_eye_data()
        
        # Check if files were created
        files = os.listdir(self.test_dir)
        self.assertTrue(any(f.endswith('.csv') for f in files))
        
        # Test data clearing
        self.assertEqual(len(self.tracker.eye_data['timestamp']), 0)

class TestPyEyeTrackPerformance(TestPyEyeTrackBase):
    """Test performance monitoring and metrics"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner()
        self.tracker.start_time = time.time()

    def test_performance_metrics(self):
        """Test performance monitoring functionality"""
        # Simulate frame processing
        for _ in range(10):
            time.sleep(0.1)  # Simulate processing time
            self.tracker.update_performance_metrics()
        
        self.assertGreater(len(self.tracker.frame_times), 0)
        self.assertIsNotNone(self.tracker.last_frame_time)
        self.assertGreaterEqual(self.tracker.current_fps, 0)

    def test_recording_stats(self):
        """Test recording statistics calculation"""
        self.tracker.total_frames = 100
        self.tracker.dropped_frames = 5
        
        stats = self.tracker._calculate_recording_stats(
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        self.assertIn('session_info', stats)
        self.assertIn('performance', stats)
        self.assertIn('data_quality', stats)
        
        self.assertEqual(stats['performance']['total_frames'], 100)
        self.assertEqual(stats['performance']['dropped_frames'], 5)

class TestPyEyeTrackCamera(TestPyEyeTrackBase):
    """Test camera handling and initialization"""
    
    @patch('cv2.VideoCapture')
    def test_camera_initialization(self, mock_cv2_cap):
        """Test camera initialization"""
        mock_cv2_cap.return_value = self.mock_cap
        
        tracker = PyEyeTrackRunner()
        success = tracker.initialize_camera(0)
        
        self.assertTrue(success)
        self.assertEqual(tracker.current_camera, 0)

    @patch('cv2.VideoCapture')
    def test_camera_switching(self, mock_cv2_cap):
        """Test camera switching"""
        mock_cv2_cap.return_value = self.mock_cap
        
        tracker = PyEyeTrackRunner()
        tracker.initialize_camera(0)
        
        success = tracker.switch_camera(1)
        self.assertTrue(success)
        self.assertEqual(tracker.current_camera, 1)
        
        # Test invalid camera
        with self.assertRaises(ValueError):
            tracker.switch_camera(-1)

class TestPyEyeTrackPupilDetection(TestPyEyeTrackBase):
    """Test pupil detection functionality"""
    
    @patch('cv2.VideoCapture')
    def test_pupil_detection(self, mock_cv2_cap):
        """Test pupil detection"""
        mock_cv2_cap.return_value = self.mock_cap
        
        pupil_tracker = PupilTracking(source=0)
        frame, pupils = pupil_tracker.detect_pupil(self.test_image)
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, self.test_image.shape)

    def test_pupil_size_calculation(self):
        """Test pupil size calculation"""
        pupil_tracker = PupilTracking(source=0)
        
        # Create test eye region
        eye_region = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(eye_region, (25, 25), 5, 255, -1)  # Add pupil
        
        size = pupil_tracker.calculate_pupil_size(eye_region)
        self.assertGreater(size, 0)

class TestPyEyeTrackRecording(TestPyEyeTrackBase):
    """Test recording functionality"""
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session"
        )
        self.tracker.data_dir = self.test_dir

    def test_recording_controls(self):
        """Test recording controls"""
        # Test start recording
        self.tracker.start_recording()
        self.assertTrue(self.tracker.recording)
        self.assertIsNotNone(self.tracker.start_time)
        
        # Test stop recording
        self.tracker.stop_recording()
        self.assertFalse(self.tracker.recording)

    def test_markers_and_notes(self):
        """Test markers and notes functionality"""
        # Add marker
        self.tracker.add_marker("Test marker")
        self.assertEqual(len(self.tracker.markers), 1)
        self.assertEqual(self.tracker.markers[0]['note'], "Test marker")
        
        # Add note
        self.tracker.add_note("Test note")
        self.assertEqual(len(self.tracker.notes), 1)
        self.assertEqual(self.tracker.notes[0]['note'], "Test note")

@unittest.skipIf(not AUDIO_AVAILABLE, "PyAudio not available")
class TestPyEyeTrackAudioVideo(TestPyEyeTrackBase):
    """Test audio and video recording"""
    
    def setUp(self):
        """Set up each test"""
        self.video_recorder = VideoRecorder(file_name="test_video")
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