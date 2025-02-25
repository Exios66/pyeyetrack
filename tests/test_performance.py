import unittest
import tempfile
import shutil
import time
from datetime import datetime
import numpy as np
from pyEyeTrack.PyEyeTrackRunnerClass import PyEyeTrackRunner

class TestPyEyeTrackPerformance(unittest.TestCase):
    """Test performance monitoring and metrics"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        cls.test_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests"""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test"""
        self.tracker = PyEyeTrackRunner(
            participant_id="test_participant",
            session_id="test_session"
        )
        self.tracker.data_dir = self.test_dir
        self.tracker.start_time = time.time()
    
    def test_performance_metrics(self):
        """Test performance monitoring functionality"""
        # Simulate frame processing
        for _ in range(10):
            time.sleep(0.01)  # Simulate processing time
            self.tracker.update_performance_metrics()
        
        self.assertGreater(len(self.tracker.frame_times), 0)
        self.assertIsNotNone(self.tracker.last_frame_time)
        self.assertGreaterEqual(self.tracker.current_fps, 0)
        
        # Test FPS calculation
        fps = len(self.tracker.frame_times) / sum(self.tracker.frame_times)
        self.assertAlmostEqual(fps, self.tracker.current_fps, places=1)
    
    def test_recording_stats(self):
        """Test recording statistics calculation"""
        # Set up test data
        self.tracker.total_frames = 100
        self.tracker.dropped_frames = 5
        self.tracker.current_camera = 0
        
        # Add some test eye tracking data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        for i in range(10):
            self.tracker.eye_data['timestamp'].append(timestamp)
            self.tracker.eye_data['data_quality'].append(0.8 + i * 0.02)  # Varying quality
            self.tracker.eye_data['left_blink'].append(1 if i % 3 == 0 else 0)  # Some blinks
            self.tracker.eye_data['right_blink'].append(1 if i % 4 == 0 else 0)
            self.tracker.eye_data['gaze_x'].append(500 + i * 10)
            self.tracker.eye_data['gaze_y'].append(300 + i * 5)
        
        # Calculate stats
        stats = self.tracker._calculate_recording_stats(
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        # Test session info
        self.assertIn('session_info', stats)
        self.assertEqual(stats['session_info']['participant_id'], "test_participant")
        self.assertEqual(stats['session_info']['session_id'], "test_session")
        
        # Test performance metrics
        self.assertIn('performance', stats)
        self.assertEqual(stats['performance']['total_frames'], 100)
        self.assertEqual(stats['performance']['dropped_frames'], 5)
        self.assertEqual(stats['performance']['camera_id'], 0)
        self.assertGreaterEqual(stats['performance']['frame_rate'], 0)
        
        # Test data quality metrics
        self.assertIn('data_quality', stats)
        self.assertGreater(stats['data_quality']['average_quality_score'], 0)
        self.assertGreater(stats['data_quality']['total_blinks'], 0)
        self.assertGreater(stats['data_quality']['valid_gaze_points'], 0)
        self.assertGreaterEqual(stats['data_quality']['data_completeness'], 0)
        self.assertLessEqual(stats['data_quality']['data_completeness'], 1)
    
    def test_frame_time_tracking(self):
        """Test frame time tracking"""
        # Test initial state
        self.assertEqual(len(self.tracker.frame_times), 0)
        self.assertIsNone(self.tracker.last_frame_time)
        
        # Add some frame times
        self.tracker.update_performance_metrics()
        time.sleep(0.01)
        self.tracker.update_performance_metrics()
        
        # Test frame time tracking
        self.assertEqual(len(self.tracker.frame_times), 1)
        self.assertIsNotNone(self.tracker.last_frame_time)
        
        # Test frame time buffer limit
        for _ in range(150):  # More than buffer size
            time.sleep(0.001)
            self.tracker.update_performance_metrics()
        
        self.assertLessEqual(len(self.tracker.frame_times), 100)  # Buffer limit
    
    def test_fps_calculation(self):
        """Test FPS calculation"""
        # Test with consistent frame times
        frame_time = 0.033  # ~30 FPS
        for _ in range(10):
            time.sleep(frame_time)
            self.tracker.update_performance_metrics()
        
        expected_fps = 1 / np.mean(self.tracker.frame_times)
        self.assertAlmostEqual(self.tracker.current_fps, expected_fps, delta=5)
        
        # Test FPS update interval
        self.assertGreaterEqual(time.time() - self.tracker.last_fps_update,
                              self.tracker.fps_update_interval)

if __name__ == '__main__':
    unittest.main() 