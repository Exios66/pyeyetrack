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
        
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            # Set directory permissions (rwxr-xr-x)
            os.chmod(self.session_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
            logger.debug(f"Created test session directory with proper permissions: {self.session_dir}")
        except Exception as e:
            logger.error(f"Failed to create/set permissions for session directory: {e}")
            raise
        
        # Create sample eye tracking data
        self.test_filename = f'eye_tracking_data_{self.session_timestamp}.csv'
        logger.debug(f"Test filename: {self.test_filename}")
        
        self.sample_data = {
            'Session_ID': ['test_session_001'] * 5,
            'Time': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            ],
            'Left_Pupil_X': [100, 102, 103, 101, 100],
            'Left_Pupil_Y': [150, 152, 153, 151, 150],
            'Right_Pupil_X': [200, 202, 203, 201, 200],
            'Right_Pupil_Y': [250, 252, 253, 251, 250]
        }
        
        # Save test data in both directories
        # 1. Save to main output directory (for actual testing)
        self.test_csv_path = os.path.join(self.output_dir, self.test_filename)
        # 2. Save to test session directory (for preservation)
        self.preserved_csv_path = os.path.join(self.session_dir, self.test_filename)
        
        try:
            df = pd.DataFrame(self.sample_data)
            # Save both copies
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
                'run_as_root': os.geteuid() == 0
            }
            metadata_path = os.path.join(self.session_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            # Set metadata file permissions (rw-r--r--)
            os.chmod(metadata_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            logger.debug(f"Saved test metadata with proper permissions to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to write files: {e}")
            raise
        
    def tearDown(self):
        """Clean up after each test - remove only the test file from output directory"""
        logger.debug("Starting tearDown")
        try:
            if os.path.exists(self.test_csv_path):
                os.remove(self.test_csv_path)
                logger.debug(f"Removed test file from output directory: {self.test_csv_path}")
            else:
                logger.warning(f"Test file not found during cleanup: {self.test_csv_path}")
                
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
        # Log current directory contents
        logger.debug(f"Current directory: {os.getcwd()}")
        logger.debug(f"Output directory contents: {os.listdir(self.output_dir)}")
        
        self.assertTrue(os.path.exists(self.test_csv_path), "CSV file was not created")
        self.assertTrue(os.path.exists(self.output_dir), "Output directory does not exist")
        self.assertIn(self.test_filename, os.listdir(self.output_dir), "CSV file not found in output directory")
        
    def test_output_directory_structure(self):
        """Test that the output directory exists and has the correct structure"""
        # Log directory information
        logger.debug(f"Testing directory: {self.output_dir}")
        logger.debug(f"Directory exists: {os.path.exists(self.output_dir)}")
        logger.debug(f"Is directory: {os.path.isdir(self.output_dir)}")
        logger.debug(f"Directory permissions: {oct(os.stat(self.output_dir).st_mode)}")
        
        # Check output directory exists
        self.assertTrue(os.path.exists(self.output_dir), "Output directory does not exist")
        self.assertTrue(os.path.isdir(self.output_dir), "Output path is not a directory")
        
        # Check output directory permissions
        self.assertTrue(os.access(self.output_dir, os.W_OK), "Output directory is not writable")
        self.assertTrue(os.access(self.output_dir, os.R_OK), "Output directory is not readable")
        
        # Check CSV file is in the output directory
        files = os.listdir(self.output_dir)
        logger.debug(f"Files in output directory: {files}")
        
        csv_files = [f for f in files if f.endswith('.csv')]
        logger.debug(f"CSV files found: {csv_files}")
        
        self.assertGreater(len(csv_files), 0, "No CSV files found in output directory")
        self.assertIn(self.test_filename, csv_files, "Test CSV file not found in output directory")
        
    def test_csv_file_structure(self):
        """Test that CSV file has the correct columns and data types"""
        df = pd.read_csv(self.test_csv_path)
        
        # Check required columns exist
        required_columns = [
            'Session_ID',
            'Time',
            'Left_Pupil_X',
            'Left_Pupil_Y',
            'Right_Pupil_X',
            'Right_Pupil_Y'
        ]
        for column in required_columns:
            self.assertIn(column, df.columns, f"Missing required column: {column}")
            
        # Check data types
        self.assertTrue(df['Left_Pupil_X'].dtype in ['float64', 'int64'], "Invalid data type for Left_Pupil_X")
        self.assertTrue(df['Left_Pupil_Y'].dtype in ['float64', 'int64'], "Invalid data type for Left_Pupil_Y")
        self.assertTrue(df['Right_Pupil_X'].dtype in ['float64', 'int64'], "Invalid data type for Right_Pupil_X")
        self.assertTrue(df['Right_Pupil_Y'].dtype in ['float64', 'int64'], "Invalid data type for Right_Pupil_Y")
        
    def test_csv_data_validity(self):
        """Test that CSV data is within valid ranges"""
        df = pd.read_csv(self.test_csv_path)
        
        # Check coordinate ranges (assuming coordinates are in pixels and can't be negative)
        self.assertTrue((df['Left_Pupil_X'] >= 0).all(), "Invalid negative X coordinate for left pupil")
        self.assertTrue((df['Left_Pupil_Y'] >= 0).all(), "Invalid negative Y coordinate for left pupil")
        self.assertTrue((df['Right_Pupil_X'] >= 0).all(), "Invalid negative X coordinate for right pupil")
        self.assertTrue((df['Right_Pupil_Y'] >= 0).all(), "Invalid negative Y coordinate for right pupil")
        
        # Check for reasonable maximum values (assuming 4K resolution max)
        max_resolution = 4096
        self.assertTrue((df['Left_Pupil_X'] <= max_resolution).all(), "X coordinate exceeds maximum resolution")
        self.assertTrue((df['Left_Pupil_Y'] <= max_resolution).all(), "Y coordinate exceeds maximum resolution")
        self.assertTrue((df['Right_Pupil_X'] <= max_resolution).all(), "X coordinate exceeds maximum resolution")
        self.assertTrue((df['Right_Pupil_Y'] <= max_resolution).all(), "Y coordinate exceeds maximum resolution")
        
    def test_timestamp_format(self):
        """Test that timestamps in CSV are in correct format"""
        df = pd.read_csv(self.test_csv_path)
        
        # Check timestamp format (should be YYYY-MM-DD HH:MM:SS.ffffff)
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}'
        self.assertTrue(df['Time'].str.match(timestamp_pattern).all(), "Invalid timestamp format")
        
        # Check that timestamps are in chronological order
        timestamps = pd.to_datetime(df['Time'])
        self.assertTrue((timestamps.diff()[1:] >= pd.Timedelta(0)).all(), "Timestamps are not in chronological order")
        
    def test_session_id_consistency(self):
        """Test that session ID remains consistent throughout the recording"""
        df = pd.read_csv(self.test_csv_path)
        
        # Check that session ID is consistent
        unique_session_ids = df['Session_ID'].unique()
        self.assertEqual(len(unique_session_ids), 1, "Multiple session IDs found in single recording")
        
    def test_data_continuity(self):
        """Test that there are no large unexplained jumps in pupil positions"""
        df = pd.read_csv(self.test_csv_path)
        
        # Check for continuity in pupil positions (no jumps larger than 50 pixels)
        max_jump = 50
        for col in ['Left_Pupil_X', 'Left_Pupil_Y', 'Right_Pupil_X', 'Right_Pupil_Y']:
            jumps = abs(df[col].diff())
            self.assertTrue((jumps[1:] <= max_jump).all(), f"Large unexpected jump detected in {col}")

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