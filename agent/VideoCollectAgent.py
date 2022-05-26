import os
import cv2
import sys
import json
import time
import random
import base64
import datetime
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from agent import LaprasAgent

class VideoCollectAgent(LaprasAgent.LaprasAgent):
    def __init__(self, agent_name='VideoCollectAgent', place_name='Robot'):
        super().__init__(agent_name, place_name)
        # self.start_ts = self.curr_timestamp()
        self.subscribe(f'Robot/context/RobotDetectedImage')

        self.dir = 'video/'
        self.out_fname = 'out.mp4'
        self.timer_cnt = 0
        time_interval = 1
        self.create_timer(self.timer_callback, timer_period=time_interval)
        self.images = []
        self.ts = []

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        # print('arrived')
        if dict.get('name') == 'RobotDetectedImage':
            
            img_str = dict['value']
            imgdata = base64.b64decode(img_str)
            cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
            self.images.append(cv_img)
            self.ts.append(time.time())
    
    def timer_callback(self):
        self.timer_cnt += 1
        runtime = self.ts[-1]-self.ts[0] if len(self.images) > 0 else 0
        if len(self.images) > 0:
            print(f'Collected {len(self.images)} images | Runtime: {runtime:.2f} s')

        if runtime > 60:
            height, width, layers = self.images[0].shape
            images = self.images[:]
            
            size = width, height
            fps = len(images) / runtime
            print(height, width, layers, fps)
            
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(self.dir+now+'.mp4', cv2.VideoWriter_fourcc(*'FMP4'), fps, size)
            for frame in images:
                out.write(frame)
            out.release()

            # self.loop_stop()
            self.disconnect()
        

if __name__ == '__main__':
    client = VideoCollectAgent()
    client.loop_forever()
    client.disconnect()