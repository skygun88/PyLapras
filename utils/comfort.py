import os
import sys
import numpy as np
from typing import Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)).split('PyLapras')[0]+'PyLapras')
from utils.configure import *


class PMV:
    def __init__(self, clothing_index=0.5, temperature=20, humidity=20, metabolism_index=1.0, fan=False):
        self.clothing_index = clothing_index # rest=0.5, leisure=0.67, physical workout=0.36
        self.wind_speed = 1.04/3 if fan else 0.0000001
        self.temperature = temperature
        self.mean_radiant_temperature = 27
        self.humidity = humidity
        self.metabolism_index = metabolism_index # rest=1.0, leisure=1.3, physical workout=1.8
        self.boundary = [-3, 3]


    def FNPS(self, T):
        return (np.e)**(16.6536-(4030.183/(T+235)))

    def calculatePMV(self):
        PA = self.humidity * 10 * self.FNPS(self.temperature)
        # PA = (0.61094*np.e**(17.625*self.temperature/(self.temperature+243.04)))*1000
        # print(PA)
        ICL = self.clothing_index * 0.155
        M = self.metabolism_index * 58.15
        W = 0.0000001 * 58.15
        MW = M - W
        if ICL < 0.078:
            FCL = 1 + 1.29*ICL
        else:
            FCL = 1.05 + 0.645*ICL

        HCF = 12.1*self.wind_speed

        TAA = self.temperature + 273
        TRA = self.mean_radiant_temperature + 273

        TCLA = TAA + (35.5-self.temperature) / (3.5*(6.45*ICL + 0.1))

        P1 = ICL * FCL
        P2 = P1 * 3.96
        P3 = P1 * 100
        P4 = P1 * TAA
        P5 = 308.7 - 0.028 * MW + P2 * ((TRA/100)**4)
        XN = TCLA / 100
        XF = XN
        
        
        N=0
        EPS = 0.00015
        
        while True:
            XF = (XF+XN)/2
            HCN = 2.38 * abs(100*XF - TAA)**0.25

            if HCF > HCN:
                HC = HCF
            else:
                HC = HCN

            XN = (P5+P4*HC - P2*(XF**4)) / (100 + P3*HC)
            N += 1

            if N > 150:
                return 99999
            if abs(XN-XF) <= EPS:
               break

        TCL = 100*XN - 273
        HL1 = 3.05 * 0.001 * (5733 - 6.99*MW - PA)
        if MW > 58.15:
            HL2 = 0.42*(MW-58.15)
        else:
            HL2 = 0 # "0!" ?

        HL3 = 1.7 * 0.00001 * M * (5867-PA)
        HL4 = 0.0014 * M * (34-self.temperature)
        HL5 = 3.96*FCL*((XN**4) - (TRA/100)**4)
        HL6 = FCL*HC*(TCL-self.temperature)

        TS = 0.303 * np.e**(-0.036*M) + 0.028
        PMV = TS * (MW-HL1-HL2-HL3-HL4-HL5-HL6)
        return PMV

def calculate_BMR(age, gender):
    m_man = 76.2
    l_man = 1.7435
    m_woman = 57.98
    l_woman = 1.6177
    if gender == 0:
        m = m_man
        l = l_man
    else: 
        m = m_woman
        l = l_woman
    
    BMR = 58*m + 1741*l - 14*age -470*gender + 227
    BMR_W = BMR / (3.6 * 24)
    return BMR_W + 19.4

def calculate_CFB(BMR_individual):
    BMR_standard = 114.6
    CFB = BMR_individual / BMR_standard
    return CFB

def calculate_MET(act, age, gender):
    act_to_met = {0: 1.0, 1: 1.2, 2:1.9}
    MET_act = act_to_met[act]
    BMR_individual = calculate_BMR(age, gender)
    CFB = calculate_CFB(BMR_individual)
    MET = MET_act * CFB
    return MET 

def calculate_CLI(top, bottom):
    top_CLO = 0.19 if top == 0 else 0.25
    top_CLO += 0.08 + 0.03
    bottom_CLO = 0.08 if bottom == 0 else 0.24
    bottom_CLO += 0.02 + 0.02
    return top_CLO + bottom_CLO + 0.15

def estimate_thermal_comfort(temperature: float, humidity:float, fan: bool, har: int=0, clothing: Tuple[int, int]=[0, 0], age: int=28, gender: int=0):
    MET_per = calculate_MET(har, age, gender)
    CLI = calculate_CLI(clothing[0], clothing[1])
    theraml_comfort = PMV(CLI, temperature, humidity, MET_per, fan).calculatePMV()
    return theraml_comfort

def clothing_analysis(clothing_list):
    top, bottom = 0, 0
    if 1 in (clothing_list[LONG_SLEEVE_TOP], clothing_list[LONG_SLEEVE_OUTWEAR], clothing_list[LONG_SLEEVE_DRESS]):
        top = 1
    if 1 in (clothing_list[TROUSERS], clothing_list[LONG_SLEEVE_DRESS]):
        bottom = 1
    return top, bottom

