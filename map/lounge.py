import os
import pandas as pd

PYLAPRAS_PATH = os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras'

def load_map():
    df = pd.loa