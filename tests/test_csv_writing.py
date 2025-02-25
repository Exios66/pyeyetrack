import unittest
import os
import pandas as pd
from datetime import datetime
import shutil
import time
import json
import logging
import sys
import stat
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestCSVWriting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Check for root privileges
        if os.geteuid() != 0:
            logger.warning("Tests should be run with sudo privileges")
            
        # Get absolute path for output directory
        cls.output_dir = os.path.abspath("Output")
        cls.test_data_dir = os.path.join(cls.output_dir, "test_data")
        
        # Create output and test data directories with proper permissions
        for directory in [cls.output_dir, cls.test_data_dir]:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # Set directory permissions (rwxr-xr-x)
                os.chmod(directory, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            except Exception as e:
                logger.error(f"Failed to create/set permissions for directory {directory}: {e}")
                raise
                
        logger.debug(f"Created/verified directories with proper permissions: Output={cls.output_dir}, TestData={cls.test_data_dir}")

    def setUp(self):
        """Set up test environment before each test"""
        # Create unique test session directory
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.test_data_dir, f"test_session_{self.session_timestamp}")
        
        # Define all required subdirectories
        subdirs = [
            'data/raw_data',
            'data/processed_data',
            'metadata',
            'logs',
            'exports'
        ]
        
        try:
            # Create session directory
            os.makedirs(self.session_dir, exist_ok=True)
            os.chmod(self.session_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            
            # Create all subdirectories with proper permissions
            for subdir in subdirs:
                dir_path = os.path.join(self.session_dir, subdir)
                os.makedirs(dir_path, exist_ok=True)
                os.chmod(dir_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            
            logger.debug(f"Created test session directory with proper permissions: {self.session_dir}")
        except Exception as e:
            logger.error(f"Failed to create/set permissions for session directory: {e}")
            raise
        
        # Create sample eye tracking data
        self.test_filename = f'eye_tracking_data_{self.session_timestamp}.csv'
        logger.debug(f"Test filename: {self.test_filename}")
        
        # Create comprehensive test data
        self.sample_data = {
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') for _ in range(5)],
            'frame_number': list(range(1, 6)),
            'left_eye_x': [100, 102, 103, 101, 100],
            'left_eye_y': [150, 152, 153, 151, 150],
            'right_eye_x': [200, 202, 203, 201, 200],
            'right_eye_y': [250, 252, 253, 251, 250],
            'left_pupil_size': [5.0, 5.1, 5.0, 4.9, 5.0],
            'right_pupil_size': [5.0, 5.1, 5.0, 4.9, 5.0],
            'left_blink': [0, 0, 1, 0, 0],
            'right_blink': [0, 0, 1, 0, 0],
            'gaze_x': [350, 352, 353, 351, 350],
            'gaze_y': [400, 402, 403, 401, 400],
            'head_pose_x': [10, 11, 10, 9, 10],
            'head_pose_y': [20, 21, 20, 19, 20],
            'head_pose_z': [30, 31, 30, 29, 30],
            'data_quality': [1.0, 0.9, 0.8, 0.9, 1.0]
        }
        
        # Save test data in both directories
        # 1. Save to main output directory (for actual testing)
        self.test_csv_path = os.path.join(self.output_dir, self.test_filename)
        # 2. Save to test session directory (for preservation)
        self.preserved_csv_path = os.path.join(self.session_dir, "data", "raw_data", self.test_filename)
        
        try:
            df = pd.DataFrame(self.sample_data)
            # Create directories and save both copies
            os.makedirs(os.path.dirname(self.preserved_csv_path), exist_ok=True)
            df.to_csv(self.test_csv_path, index=False)
            df.to_csv(self.preserved_csv_path, index=False)
            
            # Set file permissions (rw-r--r--)
            for file_path in [self.test_csv_path, self.preserved_csv_path]:
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                
            logger.debug(f"Wrote CSV files with proper permissions to:\n  Test: {self.test_csv_path}\n  Preserved: {self.preserved_csv_path}")
            
            # Save test metadata
            metadata = {
                'timestamp': self.session_timestamp,
                'test_file': self.test_filename,
                'sample_size': len(df),
                'columns': list(df.columns),
                'session_info': {
                    'participant_id': 'test_participant',
                    'session_id': 'test_session',
                    'session_type': 'pilot'
                }
            }
            metadata_path = os.path.join(self.session_dir, 'metadata', 'session_metadata.json')
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            os.chmod(metadata_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            logger.debug(f"Saved test metadata with proper permissions to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to write files: {e}")
            raise
        
    def tearDown(self):
        """Clean up after each test"""
        logger.debug("Starting tearDown")
        try:
            # Remove test file from output directory
            if os.path.exists(self.test_csv_path):
                os.remove(self.test_csv_path)
                logger.debug(f"Removed test file from output directory: {self.test_csv_path}")
            
            logger.debug(f"Preserved test data remains in: {self.preserved_csv_path}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    @classmethod
    def get_preserved_test_data(cls, session_timestamp=None):
        """Utility method to retrieve preserved test data"""
        if session_timestamp:
            session_dir = os.path.join(cls.test_data_dir, f"test_session_{session_timestamp}")
            if os.path.exists(session_dir):
                return [f for f in os.listdir(session_dir) if f.endswith('.csv')]
        return [d for d in os.listdir(cls.test_data_dir) if os.path.isdir(os.path.join(cls.test_data_dir, d))]

    def test_csv_file_creation(self):
        """Test that CSV file exists with correct path and name"""
        self.assertTrue(os.path.exists(self.test_csv_path), "CSV file was not created")
        self.assertTrue(os.path.exists(self.preserved_csv_path), "Preserved CSV file was not created")
        
        # Verify file permissions
        test_perms = stat.S_IMODE(os.stat(self.test_csv_path).st_mode)
        preserved_perms = stat.S_IMODE(os.stat(self.preserved_csv_path).st_mode)
        self.assertEqual(test_perms, 0o644, "Incorrect test file permissions")
        self.assertEqual(preserved_perms, 0o644, "Incorrect preserved file permissions")

    def test_directory_structure(self):
        """Test directory structure and permissions"""
        # Check main directories exist
        self.assertTrue(os.path.exists(self.output_dir), "Output directory missing")
        self.assertTrue(os.path.exists(self.test_data_dir), "Test data directory missing")
        self.assertTrue(os.path.exists(self.session_dir), "Session directory missing")
        
        # Check subdirectories exist
        required_subdirs = ['data/raw_data', 'metadata', 'logs']
        for subdir in required_subdirs:
            dir_path = os.path.join(self.session_dir, subdir)
            self.assertTrue(os.path.exists(dir_path), f"Missing subdirectory: {subdir}")
            
            # Check directory permissions
            perms = stat.S_IMODE(os.stat(dir_path).st_mode)
            self.assertEqual(perms, 0o755, f"Incorrect permissions for {subdir}")

    def test_data_validation(self):
        """Test data validation and quality checks"""
        df = pd.read_csv(self.preserved_csv_path)
        
        # Check required columns
        required_columns = [
            'timestamp', 'frame_number', 
            'left_eye_x', 'left_eye_y', 
            'right_eye_x', 'right_eye_y'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        
        # Check data types
        numeric_columns = [
            'left_eye_x', 'left_eye_y', 
            'right_eye_x', 'right_eye_y',
            'left_pupil_size', 'right_pupil_size'
        ]
        for col in numeric_columns:
            self.assertTrue(np.issubdtype(df[col].dtype, np.number), 
                          f"Column {col} is not numeric")
        
        # Check value ranges
        for col in ['left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y']:
            self.assertTrue((df[col] >= 0).all(), f"Negative values in {col}")
            
        # Check quality scores
        self.assertTrue((df['data_quality'] >= 0).all() and 
                       (df['data_quality'] <= 1).all(),
                       "Invalid quality scores")

    def test_metadata_handling(self):
        """Test metadata file creation and content"""
        metadata_path = os.path.join(self.session_dir, 'metadata', 'session_metadata.json')
        self.assertTrue(os.path.exists(metadata_path), "Metadata file missing")
        
        # Check metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        required_fields = ['timestamp', 'test_file', 'sample_size', 'columns']
        for field in required_fields:
            self.assertIn(field, metadata, f"Missing metadata field: {field}")
            
        # Check session info
        self.assertIn('session_info', metadata)
        session_info = metadata['session_info']
        self.assertEqual(session_info['session_type'], 'pilot')
        self.assertEqual(session_info['participant_id'], 'test_participant')
        self.assertEqual(session_info['session_id'], 'test_session')

    def test_data_consistency(self):
        """Test data consistency between files"""
        test_df = pd.read_csv(self.test_csv_path)
        preserved_df = pd.read_csv(self.preserved_csv_path)
        
        # Compare dataframes
        pd.testing.assert_frame_equal(test_df, preserved_df, 
                                    "Data mismatch between test and preserved files")
        
        # Verify data matches original sample
        for col in self.sample_data:
            self.assertTrue(all(test_df[col] == self.sample_data[col]),
                          f"Data mismatch in column {col}")

if __name__ == '__main__':
    # Check if running with sudo
    if os.geteuid() != 0:
        logger.warning("Tests should be run with sudo privileges. Attempting to re-run with sudo...")
        try:
            os.execvp('sudo', ['sudo', sys.executable] + sys.argv)
        except Exception as e:
            logger.error(f"Failed to re-run with sudo: {e}")
            sys.exit(1)
    unittest.main() 