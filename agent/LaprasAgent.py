import paho.mqtt.client as mqtt
from threading import Thread
import json
import time


class LaprasAgent(mqtt.Client):
    def __init__(self, agent_name='LaprasAgent', place_name='N1CDSNLab823'):
        super().__init__()
        self.agent_name = agent_name
        self.place_name = place_name
        self.status = -1
        self.timer_list = [] # List for timer callack threads 
        self.in_loop = False # flag to terminate timer callback when loop is stop
        self.broker_address = ('smart-iot.kaist.ac.kr', 18830)
        self.connect()

    ''' Publisher for Lapras (Decided Placename & Publisher) '''
    def publish(self, type, name, msg, qos=2):
        ''' type: [context, functionality, action, task] '''
        topic = f'{self.place_name}/{type}/{name}' 
        json_msg = json.dumps(msg)
        return super().publish(topic, payload=json_msg, qos=qos, retain=False, properties=None)

    def publish_func(self, name, arguments=[], qos=2):
        msg = {
            'type': 'functionality',
            'name': name,
            'arguments': arguments,
            'publisher': self.agent_name,
            'timestamp': self.curr_timestamp()
        }
        return self.publish('functionality', name, msg, qos=qos)
    
    def publish_context(self, name, value, qos=2):
        msg = {
            'type': 'context',
            'name': name,
            'value': value,
            'publisher': self.agent_name,
            'timestamp': self.curr_timestamp()
        }
        return self.publish('context', name, msg, qos=qos)

    # def publish(self, topic, payload=None, qos=0, retain=False, properties=None): # original one
    #     return super().publish(topic, payload=payload, qos=qos, retain=retain, properties=properties)

    ''' Basic overriding '''
    def connect(self):
        return super().connect(self.broker_address[0], self.broker_address[1])

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("connected OK")
        else:
            print("Bad connection Returned code=", rc)

    def on_disconnect(self, client, userdata, flags, rc=0):
        print(str(rc))

    def on_publish(self, client, userdata, mid):
        print("In on_pub callback mid= ", mid)

    def on_subscribe(self, client, userdata, mid, granted_qos):
        print("subscribed: " + str(mid) + " " + str(granted_qos))

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        print(f'Arrived message: {dict}')


    ''' About Timer callack '''
    def create_timer(self, func, timer_period):
        t = Thread(target=self.callback_timer, args=(func, timer_period))
        t.daemon = True
        self.timer_list.append(t)

    def callback_timer(self, func, timer_period):
        while self.in_loop:
            time.sleep(timer_period)
            func()
    
    ''' About Loop start (+Timer callback start) '''
    def loop_forever(self, timeout=1, max_packets=1, retry_first_connection=False):
        self.in_loop = True
        print(f'[{self.place_name}/{self.agent_name}] Start MQTT Loop - Broker: {self.broker_address[0]}:{self.broker_address[1]}, Timers: {len(self.timer_list)}')
        ''' Start timer callbacks '''
        for t in self.timer_list:
            t.start()
        return super().loop_forever(timeout=timeout, max_packets=max_packets, retry_first_connection=retry_first_connection)

    def loop_stop(self, force=False):
        self.in_loop = False
        for t in self.timer_list: # wait to terminate all timer threads
            t.join()
        return super().loop_stop(force=force)

    def curr_timestamp(self): # return KST Timestamp (msec)
        curr_nano = time.time()
        return curr_nano//1000