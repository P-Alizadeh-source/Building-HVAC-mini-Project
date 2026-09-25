"""Generate simple synthetic building data and run a thermostat."""
import numpy as np
import pandas as pd
from building import Building

DT=3600
Q_HEAT=5000.0
Q_COOL=-4000.0

def weather(hours=168, seed=42):
    rng=np.random.default_rng(seed); h=np.arange(hours)
    T_out=10+8*np.sin(2*np.pi*h/24-np.pi/2)+rng.normal(0,1.5,hours)
    solar=np.maximum(0,3000*np.sin(2*np.pi*(h%24-6)/24))
    solar=np.where((h%24>6)&(h%24<18),solar,0)
    occupied=((h%24>7)&(h%24<19)).astype(float)
    internal=400+600*occupied+rng.normal(0,50,hours)
    return T_out,solar,internal

def thermostat(T, *args, low=20, high=24):
    if T<low: return Q_HEAT
    if T>high: return Q_COOL
    return 0.0

def electrical_power(Q, heat_cop=3.5, cool_cop=3.0, fan=300):
    if Q>0: return Q/heat_cop+fan
    if Q<0: return abs(Q)/cool_cop+fan
    return 0.0

def simulate(hours=168, controller=thermostat, seed=42):
    b=Building(); To,solar,internal=weather(hours,seed)
    Ta=Tm=21.0; rows=[]
    for k in range(hours):
        Q=float(controller(Ta,k,To,solar,internal))
        P=electrical_power(Q)
        rows.append([k,To[k],solar[k],internal[k],Ta,Tm,Q,P])
        Ta,Tm=b.step(Ta,Tm,To[k],internal[k],solar[k],Q)
    df=pd.DataFrame(rows,columns=['hour','T_out','Q_solar','Q_internal','T_in','T_mass','Q_HVAC','P_electric'])
    df['energy_kWh']=df.P_electric*DT/3.6e6
    return df

if __name__=='__main__':
    d=simulate(); print(d[['T_in','P_electric']].describe()); print('Energy:',d.energy_kWh.sum())
