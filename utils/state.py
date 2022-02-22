import json
VALUE_EMBEDDING = {'on': 1.0, 'off': 0.0}

class StateCollector:
    def __init__(self) -> None:
        self.state = dict()
        self.prev_state = dict()

    def update_state(self, msg) -> bool:
        transition = False
        dict_string = str(msg.payload.decode("utf-8"))
        dict = json.loads(dict_string)

        name = dict.get('name')
        value = dict.get('value')

        if type(value) == str:
            value = VALUE_EMBEDDING[value.lower()]
        
        self.state[name] = value
        if self.prev_state.get(name) != None and self.state[name] != self.prev_state[name]:
            transition = True
        self.prev_state[name] = value
        return transition

    def get(self, keys):
        return [self.state.get(key) for key in keys]



