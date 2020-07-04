from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
import random
import os
import sys
import threading
import subprocess
import WidgetMachine as SlotM
from datahandling import QueueHandling


class MyImageViewerWidget(QFrame):
    def __init__(self,*args):
        super(MyImageViewerWidget, self).__init__(*args)
        self.setGeometry(0, 0, 800, 600)
        self.ui = SlotM.Ui_Form()
        self.ui.setupUi(self)
        self.games_played = 0
        root_directory = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(root_directory, "slot_machine_symbols.png")
        self.px = QPixmap(path)

        self.x = [0, 0, 0, 300, 300, 300, 600, 600, 600]
        self.y = [0, 300, 600, 0, 300, 600, 0, 300, 600]

        rect = QRect(0, 0, 300, 300)
        cropped = self.px.copy(rect)
        self.ui.mLabel.setPixmap(cropped)
        self.ui.mLabel2.setPixmap(cropped)
        self.ui.mLabel3.setPixmap(cropped)
        self.queue_handler = QueueHandling()


    def select_random_image(self):
        """[This function selects an image randomly and returns it after cropping it.]

        Returns:
            [image]: [selects and returns a cropped image.]
        """
        a = random.randint(0, len(self.x) - 1)
        self.rect = QRect(self.x[a], self.y[a], 300, 300)
        cropped = self.px.copy(self.rect)
        return cropped,a


    def spin(self):
        """[This function spins the slots till three blinks are detected.
        The function accesses the queue to monitor the blinks.
        On the first blink, the first slot stops spinning.
        With every blink detection, consecutive slot stops spinning.
        The function checks if the images on all three slots are the same. If so, the user wins the jackpot. ]
        """
        blink_count = 0
        for _ in range(0, 200):
            time.sleep((50 + 25 * 9) / 1000)
            
            
            if self.queue_handler.is_empty() == False:
                queue_element = self.queue_handler.get_data() 
                if queue_element == True:
                    blink_count += 1 
        
            if blink_count < 1:

                
                cropped,a = self.select_random_image()
                self.ui.mLabel.setPixmap(cropped)

            
                cropped,b = self.select_random_image()
                self.ui.mLabel2.setPixmap(cropped)

                
                cropped,c = self.select_random_image()
                self.ui.mLabel3.setPixmap(cropped)

            elif blink_count < 2:
                
                
                cropped,b = self.select_random_image()
                self.ui.mLabel2.setPixmap(cropped)

                cropped,c = self.select_random_image()
                self.ui.mLabel3.setPixmap(cropped)

            elif blink_count < 3:

                cropped,c = self.select_random_image()
                self.ui.mLabel3.setPixmap(cropped)
                
            elif blink_count >=3:
                break
        
            QApplication.processEvents()
        
        
        self.games_played += 1
        if a == b and c == b:
            print("===============")
            print("=== JACKPOT ===")
            print("===============")

        else:
            print("Game Over!")
            self.queue_handler.add_data('Stop')
            
        
        if self.games_played >1:
            return

            


class MyMainWindow(QMainWindow):

    def __init__(self , parent=None):
        
        QWidget.__init__(self, parent=parent)
        self.setGeometry(500, 450, 940, 320)
        self.setFixedSize(940, 320)
        self.setWindowTitle('Slot Machine')

        self.mDisplay = MyImageViewerWidget(self)

    

    def keyPressEvent(self, e):
        """[This function detects a key press. 
        On the detection of key press of space bar, the spin function is called.]

        Args:
            e ([type]): [description]
        """
        if e.key() == QtCore.Qt.Key_Space:
            self.mDisplay.spin()



def main():
    """[Executes the UI]
    """
    app = QApplication(sys.argv)
    w = MyMainWindow()
    w.show()
    app.exec_()

