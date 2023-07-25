import os
import sys
from string import Template
from typing import Optional
from fastapi import FastAPI, Form
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse

import platform
from threading import Thread


sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')

from utils.configure import *
from agent.WebFeedbackCollectorAgent import WebFeedbackCollectorAgent


agent = WebFeedbackCollectorAgent()
agent_t = Thread(target=agent.loop_forever)
agent_t.daemon = True
agent_t.start()
app = FastAPI()

def float_format(f_value):
    return f'{f_value:.2f}'

def update_window(path):
    with open(path, 'r') as f:
        res = f.read()
        f.close()

    new_res = Template(res).substitute(tem0=float_format(agent.tem0), tem1=float_format(agent.tem1), 
                                    hum0=float_format(agent.hum0), hum1=float_format(agent.hum1), 
                                    ac_power=agent.ac_power, ac_temperature=agent.ac_temperature, 
                                    tem_state=float_format(agent.tem_state), hum_state=float_format(agent.hum_state), ac_state=agent.ac_state,
                                    rl_state=agent.rl_state
                                    )
    
    return new_res


@app.get("/test", response_class=HTMLResponse)
async def read_item(q: Optional[str] = None):
    new_res = update_window(path='ui.html')

    return new_res

@app.post("/test", response_class=HTMLResponse)
async def read_item(q: Optional[str] = None):
    new_res = update_window(path='ui.html')

    return new_res

@app.post("/tempUP")
async def tempUP(q: Optional[str] = None):
    agent.temp_up()

    return RedirectResponse("/test")

@app.post("/tempDOWN")
async def tempDOWN(q: Optional[str] = None):
    agent.temp_down()

    return RedirectResponse("/test")

@app.post("/tempStill")
async def tempStill(q: Optional[str] = None):
    agent.temp_still()

    return RedirectResponse("/test")

@app.post("/set_ac/")
async def set_ac(point=Form(...)):
    agent.set_ac(int(point))

    return RedirectResponse("/test")

@app.post("/PowerON/")
async def power_on():
    agent.power_on()

    return RedirectResponse("/test")

@app.post("/PowerOFF/")
async def power_off():
    agent.power_off()

    return RedirectResponse("/test")