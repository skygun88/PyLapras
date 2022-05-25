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
    def publish(self, type, name, msg, qos=2, retain=False):
        ''' type: [context, functionality, action, task] '''
        topic = f'{self.place_name}/{type}/{name}' 
        json_msg = json.dumps(msg)
        return super().publish(topic, payload=json_msg, qos=qos, retain=retain, properties=None)

    def publish_func(self, name, arguments=[], qos=2, retain=False):
        msg = {
            'type': 'functionality',
            'name': name,
            'publisher': self.agent_name,
            'timestamp': self.curr_timestamp()
        }
        if len(arguments) > 0:
            msg['arguments'] = arguments
        return self.publish('functionality', name, msg, qos=qos, retain=retain)
    
    def publish_action(self, name, arguments=[], qos=2, retain=False):
        msg = {
            'type': 'action',
            'name': name,
            'publisher': self.agent_name,
            'timestamp': self.curr_timestamp()
        }
        if len(arguments) > 0:
            msg['arguments'] = arguments
        return self.publish('action', name, msg, qos=qos, retain=retain)

    def publish_context(self, name, value, qos=2, retain=False):
        msg = {
            'type': 'context',
            'name': name,
            'valueType': java_type(value),
            'value': value,
            'publisher': self.agent_name,
            'timestamp': self.curr_timestamp()
        }
        return self.publish('context', name, msg, qos=qos, retain=retain)

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
        # print("In on_pub callback mid= ", mid)
        return

    def on_subscribe(self, client, userdata, mid, granted_qos):
        # print("subscribed: " + str(mid) + " " + str(granted_qos))
        return

    def on_message(self, client, userdata, msg):
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)
        # print(f'Arrived message: {dict}')
        return dict


    ''' About Timer callack '''
    def create_timer(self, func, timer_period):
        t = Thread(target=self.callback_timer, args=(func, timer_period))
        t.daemon = True
        self.timer_list.append(t)

    def callback_timer(self, func, timer_period):
        while self.in_loop:
            func()
            time.sleep(timer_period)
    
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
        return int(curr_nano*1000)


def java_type(data):
    str_type_map = {'<class \'int\'>': 'java.lang.Integer', '<class \'str\'>': 'java.lang.String', '<class \'float\'>': 'java.lang.Double', '<class \'bool\'>': 'java.lang.Boolean'}
    return str_type_map[str(type(data))]