import cv2
import threading
import time
import os
import sys


class VideoRecorder():
    """
    VideoRecorder class is used to record video.

    Methods:
        record()
            The function records video while ret is True.
        stop()
            The function stops recording video. 
            All the openCV objects are released.

    """

    def __init__(self, file_name='video'):
        self.open = True
        self.device_index = 0
        self.fps = 6
        self.fourcc = "MJPG"
        self.frameSize = (640, 480)
        self.file_name = file_name + ".avi"
        
        # Initialize video capture
        try:
            self.video_cap = cv2.VideoCapture(self.device_index)
            if not self.video_cap.isOpened():
                raise Exception("Failed to open camera")
            
            # Try to set camera properties
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameSize[0])
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameSize[1])
            
            self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
            self.video_out = cv2.VideoWriter(
                self.file_name,
                self.video_writer,
                self.fps,
                self.frameSize)
                
        except Exception as e:
            print(f"Error initializing video capture: {str(e)}")
            raise

    def record(self):
        """
        The function records video while ret is True. 
        Frame is written in the video every 160 ms.
        """
        while self.open:
            try:
                ret, video_frame = self.video_cap.read()
                if ret:
                    self.video_out.write(video_frame)
                    time.sleep(0.16)  # Approximately 6 FPS
                else:
                    print("Failed to capture frame")
                    break
            except Exception as e:
                print(f"Error during video recording: {str(e)}")
                break

    def stop(self):
        """
        The function stops recording video. 
        All the openCV objects are released.
        """
        if self.open:
            self.open = False
            if hasattr(self, 'video_out'):
                self.video_out.release()
            if hasattr(self, 'video_cap'):
                self.video_cap.release()
            cv2.destroyAllWindows()

    def main(self):
        """
        The function launches video recording function as a thread.
        """
        video_thread = threading.Thread(target=self.record)
        video_thread.start()
