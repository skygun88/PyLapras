import os
import re
import sys
import json
import pymongo as pm
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


def upload_replay(start_ts, ts, state):
    ''' Connect DB '''
    json_file = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))+'/resources/db_info.json'
    with open(json_file, 'r') as f:
        db_info = json.load(f)
        f.close()

    client = pm.MongoClient(db_info['address'])
    client.data.authenticate(db_info['autheticate']['name'], db_info['autheticate']['pw'], mechanism=db_info['autheticate']['mechanism'])
    db = client.data
    
    data = db.N1Lounge8F_replay

    ''' Upload Data document '''
    data.insert_one({   
        'start': start_ts,
        'timestamp': ts, 
        'state': state})
        
    client.close()

# if __name__ == '__main__':
#     print()