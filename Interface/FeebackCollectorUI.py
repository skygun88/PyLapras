import os
import sys
import cv2
import time
import math
from PyQt5.QtCore import Qt, QEvent, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QPen, QFont
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    QWidget, QGridLayout, QLineEdit, QPushButton, QApplication

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from utils.configure import *


class QFeebackCollectorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initWindow()
        self.initObjects()
        self.initLayout()  
        self.initActions()
        self.initMenus()
        self.statusbar = self.statusBar()
        
 
    def initWindow(self):
        self.setWindowTitle("AirconControlInterface")
        self.setStyleSheet("background-color:rgb(255,255,255);")
        self.setGeometry(300, 300, 600, 300)
    

    def initObjects(self):
        self.font = QFont('Arial', 12)
        self.font.setBold(True)
        ''' Buttons '''
        self.updateLabel = QLabel('Last Update: Not yet')
        self.temLabel = QLabel('Temperature: Not yet')
        self.humLabel = QLabel('Humidity: Not yet')
        self.acLabel = QLabel('A.C Mode: Not yet')

        self.labels = [self.updateLabel, self.temLabel, self.humLabel, self.acLabel]

        for i, label in enumerate(self.labels):
            label.setStyleSheet("background-color: rgba(0,0,0,0%); color: black");
            label.setFont(self.font)
            

        self.powerUpButton = QPushButton()
        self.powerUpButton.setText("Power_UP")
        self.powerDownButton = QPushButton()
        self.powerDownButton.setText("Power_DOWN")

        
        self.buttons = [self.powerUpButton, self.powerDownButton]

        for button in self.buttons:
            # button.setDisabled(True)
            button.setStyleSheet("Color: black; border-color: black; ")
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def initLayout(self):
        self.widget = QWidget(self)
        self.gridlayout = QGridLayout(self.widget)
        self.gridupper = QGridLayout()
        self.gridlower = QGridLayout()

        
        self.gridupper.addWidget(self.updateLabel, 0, 0)
        self.gridupper.addWidget(self.temLabel, 0, 1)
        self.gridupper.addWidget(self.humLabel, 0, 2)
        self.gridupper.addWidget(self.acLabel, 0, 3)

        self.gridlower.addWidget(self.powerUpButton, 0, 0)
        self.gridlower.addWidget(self.powerDownButton, 0, 1)

        self.gridlayout.addLayout(self.gridupper, 0, 0)
        self.gridlayout.addLayout(self.gridlower, 1, 0)
      
        self.setCentralWidget(self.widget)


    def initActions(self):
        self.exitAct = QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.acActs = [QAction(AC_MODE[mode], self, triggered=(lambda x: lambda: self.set_ac(x))(mode)) for mode in AC_MODE]
        self.aboutAct = QAction("&About", self, triggered=self.about)

    def initMenus(self):
        self.fileMenu = QMenu("File", self)
        self.fileMenu.addAction(self.exitAct)
        
        self.setMenu = QMenu("A.C Mode", self)
        self.setMenu.addActions(self.acActs)

        self.helpMenu = QMenu("Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.setMenu)
        self.menuBar().addMenu(self.helpMenu)

        

    def about(self):
        QMessageBox.about(self, "GUI for test RobotControlAgent",
                        """
                        ROS: https://github.com/skygun88/LG_ROS2
                        PyLapras: https://github.com/skygun88/PyLapras
                        """)


    def set_ac(self, mode):
        return True
    
    def show(self) -> None:
        result = super().show()
        return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = QFeebackCollectorUI()
    ui.show()
    sys.exit(app.exec_())