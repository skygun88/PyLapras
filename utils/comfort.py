import numpy as np

class PMV:
    def __init__(self, clothing_index=0.5, temperature=20, humidity=20, metabolism_index=1.0):
        self.clothing_index = clothing_index # rest=0.5, leisure=0.67, physical workout=0.36
        self.wind_speed = 0.0000001
        self.temperature = temperature
        self.mean_radiant_temperature = 22
        self.humidity = humidity
        self.metabolism_index = metabolism_index # rest=1.0, leisure=1.3, physical workout=1.8
        self.boundary = [-3, 3]


    def FNPS(self, T):
        return (np.e)**(16.6536-(4030.183/(T+235)))

    def calculatePMV(self):
        # PA = self.humidity * 10 * self.FNPS(self.temperature)
        PA = (0.61094*np.e**(17.625*self.temperature/(self.temperature+243.04)))*1000
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
