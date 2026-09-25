"""Simple constrained HVAC optimisation.

At each hour, test a small set of HVAC powers and choose the lowest-energy
option that keeps the next indoor temperature between 20 and 24 C.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))
import numpy as np
from building import Building
from simulation import weather, electrical_power, DT

ACTIONS=np.arange(-5000,6001,500,dtype=float)

def choose(Ta,Tm,To,solar,internal,price=1.0):
    b=Building(); options=[]
    for Q in ACTIONS:
        Tnext,Mnext=b.step(Ta,Tm,To,internal,solar,Q)
        if 20 <= Tnext <= 24:
            cost=electrical_power(Q)*DT/3.6e6*price
            options.append((cost,Q,Tnext,Mnext))
    if not options:
        # choose the action with the smallest comfort violation
        options=[(abs(b.step(Ta,Tm,To,internal,solar,Q)[0]-22),Q,*b.step(Ta,Tm,To,internal,solar,Q)) for Q in ACTIONS]
    return min(options,key=lambda x:x[0])[1]

def simulate_optimized(hours=168, seed=42, price=None):
    b=Building(); To,solar,internal=weather(hours,seed); Ta=Tm=21.; rows=[]
    for k in range(hours):
        p=1.0 if price is None else price[k]
        Q=choose(Ta,Tm,To[k],solar[k],internal[k],p)
        P=electrical_power(Q); rows.append([k,To[k],solar[k],internal[k],Ta,Tm,Q,P])
        Ta,Tm=b.step(Ta,Tm,To[k],internal[k],solar[k],Q)
    import pandas as pd
    d=pd.DataFrame(rows,columns=['hour','T_out','Q_solar','Q_internal','T_in','T_mass','Q_HVAC','P_electric'])
    d['energy_kWh']=d.P_electric*DT/3.6e6
    return d

if __name__=='__main__':
    d=simulate_optimized(); print('Energy:',d.energy_kWh.sum()); print('Comfort:',((d.T_in>=20)&(d.T_in<=24)).mean()*100)
