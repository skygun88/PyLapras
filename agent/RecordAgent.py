import os
import cv2
import sys
import json
import time
import base64
import platform
import datetime
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH") if "Linux" in platform.platform() else None
from agent.LaprasAgent import LaprasAgent
from utils.configure import *

class RobotTestAgent(LaprasAgent):
    def __init__(self, agent_name='RobotTestAgent', place_name='Robot', video_save=False, max_time=-1):
        super().__init__(agent_name, place_name)
        self.subscribe(f'{place_name}/context/RobotDetectedImage')
        self.video_dir = os.path.join(TESTER_PATH, 'video')
        self.create_timer(self.timer_callback, timer_period=1)
        self.now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_idx= 0
        self.locs = [
            (50.06, 49.3575, -50),
            (50.11, 48.36, -30),
            (50.17, 47.3, 0),
            (50.22, 46.5, 20)
        ]
        self.curr_loc = 0
        self.timer_cnt = 0
        self.start_ts = time.time()
        self.video_save = video_save
        self.frames = []
        self.temp_frames = []
        self.max_time = max_time
        self.latencies = []
        self.temp_latencies = []

        
    def move(self, x, y, z):
        self.publish_func('robotMove', arguments=[x, y, z])

    def rotate(self, angle):
        self.publish_func('robotAttend', arguments=[angle])

    def timer_callback(self):
        curr_ts = time.time()
        duration = curr_ts - self.start_ts
        
        if int(duration) % 5 == 0:
            mean_latency = sum(self.temp_latencies)/len(self.temp_latencies) if len(self.temp_latencies) > 0 else -1
            frames = len(self.temp_frames)
            temp_fps = frames/5
            print(f'Collected {self.frame_idx} frames | FPS: {temp_fps:.3f}, Latency: {mean_latency:.3f} ms')
            self.latencies.extend(self.temp_latencies)
            self.frames.extend(self.temp_frames)
            self.temp_frames.clear()
            self.temp_latencies.clear()


        if (self.max_time > 0) and (duration > self.max_time):
            if self.video_save:
                self.frames.extend(self.temp_frames)
                fps = self.frame_idx / duration
                video_path = os.path.join(self.video_dir, f'{self.now}.mp4')
                height, width, _ = self.frames[0].shape
                size = (width,height)
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

                for frame in self.frames:
                    out.write(frame)
                out.release()

            client.disconnect()
            sys.exit()

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        msg_dict = json.loads(dict_string)
        context_name = msg_dict.get('name')
        curr_ts = self.curr_timestamp()
        # print('!')
        send_ts = msg_dict['timestamp'] 

        # print(f'Send: {send_ts}, Received: {curr_ts}, Latency: {curr_ts-send_ts} ms')
        
        if context_name == 'RobotDetectedImage':
            img_str = msg_dict['value']
            imgdata = base64.b64decode(img_str)
            cv_img = cv2.imdecode(np.array(np.frombuffer(imgdata, dtype=np.uint8)) , cv2.IMREAD_COLOR)
            self.temp_frames.append(cv_img)
            if not self.video_save:
                now_path = os.path.join(self.video_dir, f'{self.now}')
                if not os.path.isdir(now_path):
                    os.makedirs(now_path)
                cv2.imwrite(os.path.join(now_path, f'{self.frame_idx}.png'), cv_img)
            self.frame_idx += 1
            self.temp_latencies.append(curr_ts-send_ts)

        else:
            print('wrong')

        

if __name__ == '__main__':
    arguments = sys.argv
    save_type = arguments[1] if len(arguments) > 1 else "img"
    max_time = int(arguments[2]) if len(arguments) > 2 else -1
    video_save = True if save_type == "video" else False 

    client = RobotTestAgent(
        video_save=video_save, 
        max_time=max_time
    )
    client.loop_forever()
    client.disconnect()