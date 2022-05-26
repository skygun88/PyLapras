import os
import sys
import time
import qimage2ndarray
from PyQt5.QtCore import Qt, QEvent, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QPalette, QPainter, QPen, QFont
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    QWidget, QGridLayout, QLineEdit, QPushButton, QApplication

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.configure import *

class CameraCommunicate(QObject):
    cameraReceive = pyqtSignal(object)

class Scroller(QScrollArea):
    def __init__(self):
        QScrollArea.__init__(self)
    def wheelEvent(self, ev):
        if ev.type() == QEvent.Wheel:
            ev.ignore()

class QRobotTesterUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initWindow()
        self.initMap()
        self.initObjects()
        self.initLayout()  
        self.initLabels()
        self.initCamera()
        self.initActions()
        self.initMenus()
        self.statusbar = self.statusBar()
        
        self.scaleFactor, self.mag_scale = 1.0, 1.0
        self.last_time_move_x, self.last_time_move_y = 0, 0
        self.target_x_map, self.target_y_map = -1, -1
        self.robot_x_map, self.robot_y_map = -1, -1
        self.last_camera_ts = -1
        self.drag = False
        self.setMouseTracking(True)

    def initWindow(self):
        self.setWindowTitle("RobotController")
        self.setStyleSheet("background-color:rgb(255,255,255);")
        self.setGeometry(300, 300, 1000, 1200)
    
    def initMap(self):
        self.ori_pixmap = QPixmap(os.path.join(MAP_PATH, 'current_map.png'))
        self.pixmap = self.ori_pixmap.copy()
        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)
        self.imageLabel.setPixmap(self.pixmap)

        self.scrollArea = Scroller()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(True)
        self.scrollArea.setAlignment(Qt.AlignCenter)
        self.scrollArea.installEventFilter(self)
        self.scrollArea.setStyleSheet("background-color:rgb(150,150,150);")
        self.scrollBarV = self.scrollArea.verticalScrollBar()
        self.scrollBarH = self.scrollArea.horizontalScrollBar()

    def initObjects(self):
        ''' Line Edits '''
        self.xLineEdit = QLineEdit()
        self.yLineEdit = QLineEdit()
        self.rLineEdit = QLineEdit()
        self.xLineEdit.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.yLineEdit.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.rLineEdit.setStyleSheet("background-color: rgb(255, 255, 255); color: rgb(0, 0, 0);")
        self.xLineEdit.setPlaceholderText('x (example: 49.5)')
        self.yLineEdit.setPlaceholderText('y (example: 47.6)')
        self.rLineEdit.setPlaceholderText('theta (-180 ~ 180)')
        
        ''' Buttons '''
        self.goButton = QPushButton()
        self.goButton.setText("Move")
        self.rotateButton = QPushButton()
        self.rotateButton.setText("Rotate")
        self.dockButton = QPushButton()
        self.dockButton.setText("Dock")
        self.undockButton = QPushButton()
        self.undockButton.setText("Undock")

        self.buttons = [self.goButton, self.rotateButton, self.dockButton, self.undockButton]

        for button in self.buttons:
            button.setDisabled(True)
            button.setStyleSheet("Color: black; border-color: black; ")

    def initLayout(self):
        self.widget = QWidget(self)
        self.gridlayout = QGridLayout(self.widget)
        self.gridupper = QGridLayout()
        
        self.gridupper.addWidget(self.xLineEdit, 0, 0)
        self.gridupper.addWidget(self.yLineEdit, 0, 1)
        self.gridupper.addWidget(self.goButton, 0, 2)
        self.gridupper.addWidget(self.rLineEdit, 0, 3)
        self.gridupper.addWidget(self.rotateButton, 0, 4)
        self.gridupper.addWidget(self.dockButton, 0, 5)
        self.gridupper.addWidget(self.undockButton, 0, 6)

        self.gridlayout.addLayout(self.gridupper, 0, 0)
        self.gridlayout.addWidget(self.scrollArea, 1, 0)
        self.setCentralWidget(self.widget)
       
    def initLabels(self):
        ''' Font '''
        self.font = QFont('Arial', 12)
        self.font.setBold(True)
                
        ''' Labels'''
        self.labels = []
        self.agentConnectLabel = QLabel('Disconnected', self.widget)
        self.labels.append(self.agentConnectLabel)
        
        self.robotStateLabel = QLabel('Uninitialized', self.widget)
        self.labels.append(self.robotStateLabel)
        
        self.robotLocationLabel = QLabel('Uninitialized', self.widget)
        self.labels.append(self.robotLocationLabel)

    
    def initCamera(self):
        ''' Camera Label'''
        self.whiteMap = QPixmap(os.path.join(TESTER_PATH, 'white.png'))
        self.cameraLabel = QLabel(self)
        self.cameraLabel.setBackgroundRole(QPalette.Base)
        self.cameraLabel.setPixmap(self.whiteMap)

        self.camera_size = self.cameraLabel.pixmap().size()/2
        self.cameraLabel.resize(self.camera_size)
        self.cameraLabel.setVisible(False)
        self.c = CameraCommunicate()
        self.c.cameraReceive.connect(self.update_camera)

        self.fpsLabel = QLabel('0.00', self)
        self.fpsLabel.resize(80, 20)
        self.fpsLabel.setStyleSheet("background-color: rgba(0,0,0,0%); color: black")
        self.fpsLabel.setAlignment(Qt.AlignLeft)
        self.fpsLabel.setVisible(False)


    def initActions(self):
        self.exitAct = QAction("Exit", self, shortcut="Ctrl+Q", triggered=self.close)
        
        self.robotConnectAct = QAction("Connect", self, triggered=self.connect_robot)

        self.poiActs = [QAction(poi, self, triggered=(lambda x: lambda: self.set_poi(x))(poi)) for poi in poi_to_location]

        self.robotDisconnectAct = QAction("Disconnect", self, triggered=self.disconnect_robot)
        self.robotDisconnectAct.setDisabled(True)
        
        self.toggleCameraAct = QAction("Camera", self, triggered=self.toggle_camera)
        self.toggleCameraAct.setCheckable(True)
        self.startRecordAct = QAction("Record Start", self, triggered=self.start_record)
        self.startRecordAct.setDisabled(True)
        self.endRecordAct = QAction("Record Stop", self, triggered=self.end_record)
        self.endRecordAct.setDisabled(True)
        self.toggleHarAct = QAction("HAR", self, triggered=self.toggle_har)
        self.toggleHarAct.setDisabled(True) 

        self.aboutAct = QAction("&About", self, triggered=self.about)

    def initMenus(self):
        self.fileMenu = QMenu("File", self)
        self.fileMenu.addAction(self.exitAct)
        
        self.robotMenu = QMenu("Robot", self)
        self.robotMenu.addAction(self.robotConnectAct)
        self.robotMenu.addSeparator()
        
        self.poiMenu = QMenu("Set PoI", self)
        self.poiMenu.addActions(self.poiActs)
        self.robotMenu.addMenu(self.poiMenu)
        self.robotMenu.addSeparator()

        self.robotMenu.addAction(self.robotDisconnectAct)
        
        self.viewMenu = QMenu("View", self)
        self.viewMenu.addAction(self.toggleCameraAct)
        self.viewMenu.addSeparator()
        
        self.recordMenu = QMenu("Record", self)
        self.recordMenu.addAction(self.startRecordAct)
        self.recordMenu.addAction(self.endRecordAct)
        self.viewMenu.addMenu(self.recordMenu)
        self.viewMenu.addSeparator()
        
        self.viewMenu.addAction(self.toggleHarAct)

        self.helpMenu = QMenu("Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.robotMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)


    def update_camera(self, img):
        qImg = qimage2ndarray.array2qimage(img)
        self.cameraLabel.setPixmap(QPixmap.fromImage(qImg))
        self.cameraLabel.update()

        if self.last_camera_ts > 0:
            fps = 1/(time.time()-self.last_camera_ts)
            self.fpsLabel.setText(f'FPS: {fps:.2f}')

        self.last_camera_ts = time.time()

    def pose_on_image(self, x, y):
        x_map = (x-self.imageLabel.frameGeometry().x()-self.scrollArea.frameGeometry().x()-self.widget.frameGeometry().x())/(self.scaleFactor*self.mag_scale)
        y_map = (y-self.imageLabel.frameGeometry().y()-self.scrollArea.frameGeometry().y()-self.widget.frameGeometry().y())/(self.scaleFactor*self.mag_scale)
        return x_map, y_map

    def pos_on_robot(self, x, y):
        x_map, y_map = self.pose_on_image(x, y)
        x_robot, y_robot = x_map/10, (1000-y_map)/10
        return x_robot, y_robot
    
    def pose_on_image_from_robot(self, x_robot, y_robot):
        x_map, y_map = x_robot*10, 1000 - y_robot*10 
        return x_map, y_map 

    def draw_target(self, x_map, y_map):
        self.pixmap = self.ori_pixmap.copy()
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
        painter.drawPoint(QPoint(x_map, y_map))        
        self.update()

    def draw_points(self):
        self.pixmap = self.ori_pixmap.copy()
        painter = QPainter(self.pixmap)
        if self.target_x_map > 0 and self.target_y_map > 0:
            painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
            painter.drawPoint(QPoint(self.target_x_map, self.target_y_map))        
        if self.robot_x_map > 0 and self.robot_y_map > 0 :
            painter.setPen(QPen(Qt.green, 3, Qt.SolidLine))
            painter.drawPoint(QPoint(self.robot_x_map, self.robot_y_map))
        self.update()

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.drawPixmap(self.rect(), self.pixmap)
        self.imageLabel.setPixmap(self.pixmap)

    def mousePressEvent(self, event): 
        ''' For map drag'''
        if event.button() == Qt.RightButton:
            self.drag = True
            return
        
        ''' Calculate mouse position on image and robot system'''
        x, y = event.x(), event.y()
        x_map, y_map = self.pose_on_image(x, y)
        x_robot, y_robot = self.pos_on_robot(x, y)
        txt = f'Mouse position: x={x_robot}, y={y_robot}'
        self.xLineEdit.setText(f'{x_robot:.2f}')
        self.yLineEdit.setText(f'{y_robot:.2f}')
        self.statusbar.showMessage(txt)
        self.target_x_map, self.target_y_map = x_map, y_map

        ''' Draw dot on mouse cursor'''
        self.draw_points()
    
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.RightButton:
            self.drag = False


    def wheelEvent(self, event):
        delta = event.angleDelta().y() /120 * 0.25 
        self.scaleImage(1.0+delta)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseMove and self.drag:
            if self.last_time_move_x == 0:
                self.last_time_move_x = event.pos().x()
            if self.last_time_move_y == 0:
                self.last_time_move_y = event.pos().y()

            distance_h = self.last_time_move_x - event.pos().x()
            distance_v = self.last_time_move_y - event.pos().y()
            self.scrollBarV.setValue(self.scrollBarV.value() + distance_v)
            self.scrollBarH.setValue(self.scrollBarH.value() + distance_h)
            self.last_time_move_x = event.pos().x()
            self.last_time_move_y = event.pos().y()
            
        elif event.type() == QEvent.MouseButtonRelease:
            self.last_time_move_x = 0
            self.last_time_move_y = 0
        return QWidget.eventFilter(self, source, event)

    def resizeEvent(self, event):
        ''' Map view Resize '''
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        max_scale = min(width, height)/1000
        self.setScale(max_scale)
        
        ''' Camera View Move '''
        self.cameraLabel.move(self.scrollArea.frameGeometry().width()-self.scrollArea.frameGeometry().x()-self.camera_size.width(), self.widget.frameGeometry().height()-self.camera_size.height())
        self.fpsLabel.move(self.cameraLabel.frameGeometry().x(), self.cameraLabel.frameGeometry().y())
        
    def about(self):
        QMessageBox.about(self, "GUI for test RobotControlAgent",
                        """
                        ROS: https://github.com/skygun88/LG_ROS2
                        PyLapras: https://github.com/skygun88/PyLapras
                        """)


    
    def toggle_camera(self):
        # print('call toogle_camera')
        self.fpsLabel.setVisible(not self.fpsLabel.isVisible())
        self.cameraLabel.setVisible(not self.cameraLabel.isVisible())
        

    def toggle_har(self):
        print('call toogle_har')
        # self.cameraLabel.setVisible(not self.cameraLabel.isVisible())

        
    def start_record(self):
        print('call start record')

    def end_record(self):
        print('call end record')
    
    def connect_robot(self):
        print('call connect robot')
        
    def disconnect_robot(self):
        print('call disconnect robot')

    def set_poi(self, poi):
        print('set poi called', poi)

    def scaleImage(self, factor):
        if self.mag_scale * factor * self.scaleFactor > 15:
            return True
        elif self.mag_scale * factor * self.scaleFactor < 0.5:
            return True
        self.mag_scale *= factor
        self.imageLabel.resize(self.scaleFactor * self.mag_scale * self.imageLabel.pixmap().size())
        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)
    
    def setScale(self, factor):
        self.scaleFactor = factor
        self.imageLabel.resize(self.scaleFactor * self.mag_scale * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

    def show(self) -> None:
        result = super().show()
        self.cameraLabel.move(self.scrollArea.frameGeometry().width()-self.scrollArea.frameGeometry().x()-self.camera_size.width(), self.widget.frameGeometry().height()-self.camera_size.height())
        for i, label in enumerate(self.labels):
            label.move(self.scrollArea.frameGeometry().x()+5, self.scrollArea.frameGeometry().y()+5+i*24)
            label.setStyleSheet("background-color: rgba(0,0,0,0%); color: black");
            label.setFont(self.font)
            label.adjustSize()
        self.fpsLabel.move(self.cameraLabel.frameGeometry().x(), self.cameraLabel.frameGeometry().y())
        return result

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = QRobotTesterUI()
    ui.show()
    sys.exit(app.exec_())