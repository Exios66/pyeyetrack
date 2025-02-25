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

class TestPyEyeTrack(unittest.TestCase):
    """Comprehensive test suite for PyEyeTrack"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources"""
        cls.test_data_dir = "test_data"
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create test image
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(cls.test_image, (320, 240), 100, (255, 255, 255), -1)
        cls.test_image_path = os.path.join(cls.test_data_dir, "test_frame.jpg")
        cv2.imwrite(cls.test_image_path, cls.test_image)
        
        # Test metadata
        cls.test_metadata = {
            "participant": {"id": "test_participant"},
            "session": {"id": "test_session"},
            "experiment": {
                "task_type": "test_task",
                "duration": 5
            }
        }

    def setUp(self):
        """Set up test fixtures"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session",
            metadata=self.test_metadata
        )
        
        # Mock camera
        self.mock_cap = Mock()
        self.mock_cap.read.return_value = (True, self.test_image)
        self.mock_cap.isOpened.return_value = True
        
    def tearDown(self):
        """Clean up after each test"""
        self.tracker.cleanup()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources"""
        import shutil
        if os.path.exists(cls.test_data_dir):
            shutil.rmtree(cls.test_data_dir)

    def test_initialization(self):
        """Test proper initialization of PyEyeTrack"""
        self.assertEqual(self.tracker.participant_id, "test_participant")
        self.assertEqual(self.tracker.session_id, "test_session")
        self.assertEqual(self.tracker.metadata, self.test_metadata)
        self.assertTrue(self.tracker.running)
        self.assertFalse(self.tracker.recording)
        self.assertFalse(self.tracker.paused)

    def test_performance_metrics(self):
        """Test performance monitoring functionality"""
        # Simulate frame processing
        for _ in range(10):
            self.tracker.update_performance_metrics()
        
        self.assertGreater(len(self.tracker.frame_times), 0)
        self.assertIsNotNone(self.tracker.last_frame_time)
        self.assertGreaterEqual(self.tracker.current_fps, 0)

    def test_data_validation(self):
        """Test eye tracking data validation"""
        test_data = {
            'left_pupil_size': 5.0,
            'gaze_x': 960,
            'gaze_y': 540,
            'head_pose_x': 45,
            'head_pose_y': 30,
            'head_pose_z': 15
        }
        
        quality_score = self.tracker.validate_eye_data(test_data)
        self.assertGreaterEqual(quality_score, 0)
        self.assertLessEqual(quality_score, 1)

    @patch('cv2.VideoCapture')
    def test_pupil_detection(self, mock_cv2_cap):
        """Test pupil detection functionality"""
        # Setup mock camera
        mock_cv2_cap.return_value = self.mock_cap
        
        # Initialize pupil tracking
        pupil_tracker = PupilTracking(source=0)
        
        # Test pupil detection
        frame, pupils = pupil_tracker.detect_pupil(self.test_image)
        
        # Verify frame was processed
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, self.test_image.shape)

    def test_data_storage(self):
        """Test data storage and retrieval"""
        # Create test directory first
        self.tracker.data_dir = os.path.join(self.test_data_dir, "data")
        os.makedirs(self.tracker.data_dir, exist_ok=True)
        
        # Add test data - ensure all arrays get the same number of elements
        timestamp = datetime.now()
        test_data = {
            'timestamp': [timestamp],
            'frame_number': [1],
            'left_eye_x': [100],
            'left_eye_y': [200],
            'right_eye_x': [300],
            'right_eye_y': [400],
            'left_pupil_size': [5.0],
            'right_pupil_size': [5.0],
            'left_blink': [0],
            'right_blink': [0],
            'gaze_x': [500],
            'gaze_y': [600],
            'head_pose_x': [10],
            'head_pose_y': [20],
            'head_pose_z': [30],
            'marker': [None],
            'data_quality': [1.0]
        }
        
        # Update tracker's eye_data with test data
        self.tracker.eye_data = test_data
        
        # Save data
        self.tracker.save_eye_data()
        
        # Verify data files were created
        self.assertTrue(os.path.exists(self.tracker.data_dir))
        
        # Look for recording directories
        recording_dirs = [d for d in os.listdir(self.tracker.data_dir) if d.startswith('recording_')]
        self.assertGreater(len(recording_dirs), 0, "No recording directories found")
        
        # Get the most recent recording directory
        latest_recording = sorted(recording_dirs)[-1]
        recording_path = os.path.join(self.tracker.data_dir, latest_recording)
        
        # Look for CSV files in the recording directory
        csv_files = [f for f in os.listdir(recording_path) if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0, "No CSV files found in recording directory")
        
        # Verify data content
        csv_path = os.path.join(recording_path, csv_files[0])
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1)  # Should have one row
        self.assertEqual(df['left_eye_x'].iloc[0], 100)
        self.assertEqual(df['left_eye_y'].iloc[0], 200)

    def test_error_handling(self):
        """Test error handling and recovery"""
        # Test invalid camera
        with self.assertRaises(ValueError):
            self.tracker.switch_camera(-1)
        
        # Test invalid data validation
        with self.assertRaises(TypeError):
            self.tracker.validate_eye_data(None)
        
        # Test invalid data directory
        self.tracker.data_dir = None
        with self.assertRaises(Exception):
            self.tracker.save_eye_data()
        
        # Test cleanup after error
        self.tracker.cleanup()
        self.assertFalse(self.tracker.running)
        
        # Test invalid frame processing
        with self.assertRaises(Exception):
            self.tracker.update_performance_metrics()
            self.tracker.frame_times = None
            self.tracker.update_performance_metrics()

    def test_calibration(self):
        """Test calibration functionality"""
        # Add calibration points
        calibration_points = [
            (100, 100),
            (100, 380),
            (540, 100),
            (540, 380)
        ]
        
        for point in calibration_points:
            self.tracker.calibration_points.append(point)
        
        self.assertEqual(len(self.tracker.calibration_points), 4)
        self.assertFalse(self.tracker.is_calibrated)

    def test_blink_detection(self):
        """Test blink detection"""
        pupil_tracker = PupilTracking(source=0)
        
        # Test with bright image (should detect blink)
        bright_eye = np.ones((50, 50), dtype=np.uint8) * 255
        self.assertTrue(pupil_tracker.detect_blink(bright_eye))
        
        # Test with dark image (should not detect blink)
        dark_eye = np.zeros((50, 50), dtype=np.uint8)
        self.assertFalse(pupil_tracker.detect_blink(dark_eye))

    def test_performance_stats(self):
        """Test performance statistics calculation and storage"""
        # Create test directory first
        self.tracker.data_dir = os.path.join(self.test_data_dir, "data")
        os.makedirs(self.tracker.data_dir, exist_ok=True)
        
        # Simulate exactly 30fps (1/30 = 0.0333... seconds per frame)
        frame_time = 1.0 / 30.0  # Exactly 30 FPS
        self.tracker.frame_times = [frame_time] * 30
        self.tracker.processing_times = [frame_time/2] * 30  # Processing time half of frame time
        self.tracker.dropped_frames = 5
        self.tracker.total_frames = 100
        
        # Save performance stats
        self.tracker.cleanup()
        
        # Verify stats file was created and contains valid data
        stats_file = os.path.join(self.tracker.data_dir, 'performance_metrics.json')
        self.assertTrue(os.path.exists(stats_file))
        
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            self.assertIn('average_fps', stats)
            self.assertIn('dropped_frames', stats)
            self.assertIn('total_frames', stats)
            self.assertIn('average_processing_time', stats)
            
            # Verify the stats values with appropriate tolerance
            self.assertAlmostEqual(stats['average_fps'], 30.0, places=1)
            self.assertEqual(stats['dropped_frames'], 5)
            self.assertEqual(stats['total_frames'], 100)
            self.assertAlmostEqual(stats['average_processing_time'], frame_time/2, places=3)

def main():
    """Run the test suite"""
    unittest.main()

if __name__ == '__main__':
    main() 