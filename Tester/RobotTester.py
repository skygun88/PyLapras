import os
import sys
from threading import Thread
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from TesterUI import QRobotTesterUI
from agent.RobotTestAgent import RobotTestAgent
from utils.configure import *

class QRobotTester(QRobotTesterUI):
    def __init__(self):
        super().__init__()
        self.initialize_handler()
        self.agent: RobotTestAgent = RobotTestAgent(self)
        self.agent_t = None

    def initialize_handler(self):
        ''' Handlers '''
        self.goButton.clicked.connect(self.robot_move)
        self.rotateButton.clicked.connect(self.robot_rotate)
        self.dockButton.clicked.connect(self.robot_dock)
        self.undockButton.clicked.connect(self.robot_undock)

    def keyPressEvent(self, e):
        if e.key() in [Qt.Key_Return, Qt.Key_Enter]:
            x_text, y_text = self.xLineEdit.text(), self.yLineEdit.text() 
            if x_text != '' and y_text != '':
                try:
                    x_robot, y_robot = float(x_text), float(y_text)
                    self.target_x_map, self.target_y_map = self.pose_on_image_from_robot(x_robot, y_robot)
                    self.draw_points()
                except:
                    self.statusbar.showMessage('wrong text maybe')
                    print('wrong text maybe')

        return super().keyPressEvent(e)

    def robot_move(self):
        ''' Blank Check '''
        x_text, y_text = self.xLineEdit.text(), self.yLineEdit.text() 
        if x_text == '' or y_text == '':
            return True
        
        ''' Draw target point to image '''
        try:
            x_robot, y_robot = float(x_text), float(y_text)
            self.target_x_map, self.target_y_map = self.pose_on_image_from_robot(x_robot, y_robot)
            self.draw_points()
        except:
            self.statusbar.showMessage('wrong text maybe')
            print('wrong text maybe')
            return True

        ''' move robot through lapras API'''
        # Check location validity
        # Send robotMove with x_robot, y_robot
        self.agent.move(x_robot, y_robot)
        self.statusbar.showMessage(f'Send message : Move to ({x_robot}, {y_robot})')
        
    
    def robot_rotate(self):
        ''' Blank Check '''
        r_text = self.rLineEdit.text()
        if r_text == '':
            return True
        try:
            angle = int(float(r_text))
        except:
            self.statusbar.showMessage('wrong text maybe')
            print('wrong angle maybe')
            return True

        ''' move robot through lapras API'''
        if angle >= -180 and angle <= 180:
            self.agent.rotate(angle)
            self.statusbar.showMessage(f'Send message : Rotate {angle} degree')
        else:
            self.statusbar.showMessage(f'Please input the angle between -180 ~ 180')

    def robot_dock(self):
        self.agent.dock()
        self.statusbar.showMessage(f'Send message : Dock')

    def robot_undock(self):
        self.agent.undock()
        self.statusbar.showMessage(f'Send message : Undock')

    def update_robot(self, connected, x_robot=-1, y_robot=-1, robot_state='UNINITIALIZED'):
        if connected:
            self.agentConnectLabel.setText('Connected')
            self.robotLocationLabel.setText(f'{x_robot:.2f}, {y_robot:.2f}')
            self.robotStateLabel.setText(f'{robot_state}')
            self.robot_x_map, self.robot_y_map = self.pose_on_image_from_robot(x_robot, y_robot)
            self.draw_points()
        else:
            self.agentConnectLabel.setText('Disconnected')
            self.robotStateLabel.setText('Disconnected')
            self.robotLocationLabel.setText('Disconnected')
            
        for label in self.labels:
            label.adjustSize()

    def initRobot(self):
        self.agentConnectLabel.setText('Connected')
        self.agentConnectLabel.adjustSize()
        self.startRecordAct.setDisabled(False)
        for button in self.buttons:
            button.setDisabled(False)
        
    def connect_robot(self):
        self.agent_t = Thread(target=self.agent.loop_forever)
        self.agent_t.daemon = True
        self.agent_t.start()
        
        ''' Button Activation '''
        self.robotConnectAct.setDisabled(True)
        self.robotDisconnectAct.setDisabled(False)
        return 

    def disconnect_robot(self):
        self.end_record()
        if self.agent_t != None:
            self.agent.disconnect()
            self.agent.loop_stop()
            self.agent_t.join()
            
            self.agentConnectLabel.setText('Disconnected')
            self.robotStateLabel.setText('Uninitialized')
            self.robotLocationLabel.setText('Uninitialized')
            
            for label in self.labels:
                label.adjustSize()
        
        ''' Reset the objects related to lapras agent '''
        self.robot_x_map, self.robot_y_map = -1, -1
        self.agent_t = None
        self.agent = RobotTestAgent(self)
        self.robotConnectAct.setDisabled(False)
        self.robotDisconnectAct.setDisabled(True)
        self.startRecordAct.setDisabled(True)
        self.endRecordAct.setDisabled(True)
        self.cameraLabel.setPixmap(self.whiteMap)
        self.fpsLabel.setText('0.00')
        for button in self.buttons:
            button.setDisabled(True)
        return 
    
    def set_poi(self, poi):
        x_robot, y_robot = poi_to_location[poi]
        self.xLineEdit.setText(f'{x_robot:.2f}')
        self.yLineEdit.setText(f'{y_robot:.2f}')
        self.target_x_map, self.target_y_map = self.pose_on_image_from_robot(x_robot, y_robot)

        self.draw_points()

    def start_record(self):
        self.agent.start_record()
        self.startRecordAct.setDisabled(True)
        self.endRecordAct.setDisabled(False)

    def end_record(self):
        self.agent.end_record()
        self.startRecordAct.setDisabled(False)
        self.endRecordAct.setDisabled(True)

    def close(self):
        self.disconnect_robot()
        return super().close()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tester = QRobotTester()
    tester.show()
    sys.exit(app.exec_())