import os
import sys
from threading import Thread
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from datetime import datetime

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from Interface.FeebackCollectorUI import QFeebackCollectorUI
from agent.FeedbackCollectorAgent import FeedbackCollectorAgent
from utils.configure import *

class QFeedbackCollector(QFeebackCollectorUI):
    def __init__(self, sensor_place='N1Lounge8F'):
        super().__init__()
        self.initialize_handler()

        self.sensor_place = sensor_place

        self.tem1, self.tem2 = None, None
        self.hum1, self.hum2 = None, None
        self.ac1, self.ac2 = None, None
        self.last_update = None
        self.ac_mode = None

        self.agent: FeedbackCollectorAgent = FeedbackCollectorAgent(self, sensor_place=self.sensor_place)
        self.agent_t = Thread(target=self.agent.loop_forever)
        self.agent_t.daemon = True
        self.agent_t.start()

        

    def initialize_handler(self):
        ''' Handlers '''
        self.powerUpButton.clicked.connect(self.aircon_power_up)
        self.powerDownButton.clicked.connect(self.aircon_power_down)

    def set_ac(self, mode):
        # self.acLabel.setText(f'A.C Mode: {AC_MODE[mode]}') 
        # self.ac_mode = mode
        self.agent.set_ac(mode)

    def update_aircon(self, ac1=None, ac2=None):
        ac_map = {'on': 1, 'off': 0}
        if ac1 != None:
            self.ac1 = ac_map[ac1.lower()]
        elif ac2 != None:
            self.ac2 = ac_map[ac2.lower()]
        else:
            print('Error Occur')
            sys.exit()
        
        print(self.ac1, self.ac2)

        if self.ac1 != None and self.ac2 != None:
            self.ac_mode = self.ac1 + self.ac2
            self.acLabel.setText(f'A.C Mode: {AC_MODE[self.ac_mode]}')
            self.update_time()


    def update_temperature(self, tem1=None, tem2=None):
        if tem1 != None:
            self.tem1 = tem1
        elif tem2 != None:
            self.tem2 = tem2
        else:
            print('Error Occur')
            sys.exit()
        
        if self.tem1 != None and self.tem2 != None:
            self.temLabel.setText(f'Temperature: {(self.tem1+self.tem2)/2:.2f}C')
            self.update_time()

    def update_humidity(self, hum1=None, hum2=None):
        if hum1 != None:
            self.hum1 = hum1
        elif hum2 != None:
            self.hum2 = hum2
        else:
            print('Error Occur')
            sys.exit()
        
        if self.hum1 != None and self.hum2 != None:
            self.humLabel.setText(f'Humidity: {(self.hum1+self.hum2)/2:.2f}%')
            self.update_time()

    def update_time(self):
        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S")
        self.updateLabel.setText(f'Last Update: {dt_string}')
    

    def aircon_power_up(self):
        if self.ac_mode == None or self.ac_mode == 2:
            print('Already Maximum power')
            return
        # self.set_ac(self.ac_mode + 1)
        self.agent.power_up()

    def aircon_power_down(self):
        if self.ac_mode == None or self.ac_mode == 0:
            print('Already Minimum power')
            return
        # self.set_ac(self.ac_mode - 1)
        self.agent.power_down()
        

    def close(self):
        self.agent.disconnect()
        self.agent.loop_stop()
        self.agent_t.join()
        return super().close()

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tester = QFeedbackCollector()
    # tester = QFeedbackCollector(sensor_place='N1SeminarRoom825')
    tester.show()
    sys.exit(app.exec_())