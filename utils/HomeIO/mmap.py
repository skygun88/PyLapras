import os
import json

with open(os.path.abspath(os.path.dirname(__file__))+'/mmap.json') as f:
    memory_map = json.load(f)
    f.close()

