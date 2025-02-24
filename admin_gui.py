from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QTextEdit, QApplication,
                             QGroupBox, QSpinBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import sys
import cv2
import time

class AdminControlPanel(QMainWindow):
    def __init__(self, tracker=None):
        super().__init__()
        self.tracker = tracker
        self.init_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(1000)  # Update every second
        
    def init_ui(self):
        self.setWindowTitle('PyEyeTrack Admin Control Panel')
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Session Info Group
        session_group = QGroupBox("Session Information")
        session_layout = QVBoxLayout()
        
        self.participant_label = QLabel("Participant: Not set")
        self.session_label = QLabel("Session ID: Not set")
        self.duration_label = QLabel("Duration: 0:00")
        self.frame_rate_label = QLabel("Frame Rate: 0 fps")
        
        session_layout.addWidget(self.participant_label)
        session_layout.addWidget(self.session_label)
        session_layout.addWidget(self.duration_label)
        session_layout.addWidget(self.frame_rate_label)
        session_group.setLayout(session_layout)
        
        # Camera Control Group
        camera_group = QGroupBox("Camera Control")
        camera_layout = QHBoxLayout()
        
        self.camera_select = QComboBox()
        self.refresh_cameras()
        self.camera_switch_btn = QPushButton("Switch Camera")
        self.camera_switch_btn.clicked.connect(self.switch_camera)
        
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_select)
        camera_layout.addWidget(self.camera_switch_btn)
        camera_group.setLayout(camera_layout)
        
        # Recording Control Group
        recording_group = QGroupBox("Recording Control")
        recording_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.add_marker_btn = QPushButton("Add Marker")
        self.add_marker_btn.clicked.connect(self.add_marker)
        
        recording_layout.addWidget(self.record_btn)
        recording_layout.addWidget(self.pause_btn)
        recording_layout.addWidget(self.add_marker_btn)
        recording_group.setLayout(recording_layout)
        
        # Statistics Group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.total_frames_label = QLabel("Total Frames: 0")
        self.dropped_frames_label = QLabel("Dropped Frames: 0")
        self.eye_detection_label = QLabel("Eye Detection Status: Not detected")
        
        stats_layout.addWidget(self.total_frames_label)
        stats_layout.addWidget(self.dropped_frames_label)
        stats_layout.addWidget(self.eye_detection_label)
        stats_group.setLayout(stats_layout)
        
        # Marker Notes Group
        notes_group = QGroupBox("Marker Notes")
        notes_layout = QVBoxLayout()
        
        self.notes_text = QTextEdit()
        self.notes_text.setPlaceholderText("Enter marker notes here...")
        self.add_note_btn = QPushButton("Add Note")
        self.add_note_btn.clicked.connect(self.add_note)
        
        notes_layout.addWidget(self.notes_text)
        notes_layout.addWidget(self.add_note_btn)
        notes_group.setLayout(notes_layout)
        
        # Add all groups to main layout
        layout.addWidget(session_group)
        layout.addWidget(camera_group)
        layout.addWidget(recording_group)
        layout.addWidget(stats_group)
        layout.addWidget(notes_group)
        
        self.update_ui_state()
        
    def refresh_cameras(self):
        """Detect available cameras"""
        self.camera_select.clear()
        camera_list = self.get_available_cameras()
        for i, name in camera_list.items():
            self.camera_select.addItem(f"{name} (ID: {i})", i)
    
    def get_available_cameras(self):
        """Get list of available camera devices"""
        camera_list = {}
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name
                if sys.platform == "darwin":  # macOS
                    name = f"Camera {i}"
                else:  # Windows/Linux
                    name = cap.getBackendName()
                camera_list[i] = name
                cap.release()
        return camera_list
    
    def switch_camera(self):
        """Switch to selected camera"""
        if self.tracker:
            camera_id = self.camera_select.currentData()
            self.tracker.switch_camera(camera_id)
    
    def toggle_recording(self):
        """Toggle recording state"""
        if self.tracker:
            if not self.tracker.recording:
                self.tracker.start_recording()
                self.record_btn.setText("Stop Recording")
            else:
                self.tracker.stop_recording()
                self.record_btn.setText("Start Recording")
    
    def toggle_pause(self):
        """Toggle pause state"""
        if self.tracker:
            self.tracker.paused = not self.tracker.paused
            self.pause_btn.setText("Resume" if self.tracker.paused else "Pause")
    
    def add_marker(self):
        """Add marker with optional note"""
        if self.tracker:
            note = self.notes_text.toPlainText().strip()
            if note:
                self.tracker.add_marker(note)
                self.notes_text.clear()
    
    def add_note(self):
        """Add a note without marker"""
        if self.tracker:
            note = self.notes_text.toPlainText().strip()
            if note:
                self.tracker.add_note(note)
                self.notes_text.clear()
    
    def update_stats(self):
        """Update statistics display"""
        if self.tracker:
            # Update session info
            self.participant_label.setText(f"Participant: {self.tracker.participant_id}")
            self.session_label.setText(f"Session ID: {self.tracker.session_id}")
            
            if self.tracker.start_time:
                duration = time.time() - self.tracker.start_time
                self.duration_label.setText(f"Duration: {int(duration//60)}:{int(duration%60):02d}")
            
            # Update statistics
            self.total_frames_label.setText(f"Total Frames: {self.tracker.total_frames}")
            self.dropped_frames_label.setText(f"Dropped Frames: {self.tracker.dropped_frames}")
            
            # Update frame rate
            if self.tracker.start_time:
                fps = self.tracker.total_frames / (time.time() - self.tracker.start_time)
                self.frame_rate_label.setText(f"Frame Rate: {fps:.1f} fps")
    
    def update_ui_state(self):
        """Update UI elements based on current state"""
        if not self.tracker:
            self.record_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.add_marker_btn.setEnabled(False)
            self.add_note_btn.setEnabled(False)
            self.camera_switch_btn.setEnabled(False)
        else:
            self.record_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.add_marker_btn.setEnabled(True)
            self.add_note_btn.setEnabled(True)
            self.camera_switch_btn.setEnabled(True)

def launch_admin_gui(tracker):
    """Launch the admin GUI"""
    app = QApplication(sys.argv)
    window = AdminControlPanel(tracker)
    window.show()
    return app, window 