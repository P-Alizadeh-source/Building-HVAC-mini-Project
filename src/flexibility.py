"""Simple time-of-use electricity-cost analysis."""
import numpy as np

def tou_price(hours):
    p=np.full(hours,0.12)
    for h in range(hours):
        if 8<=h%24<12: p[h]=0.25
        if 17<=h%24<21: p[h]=0.30
    return p

def cost(df, prices):
    return float(np.sum(df.P_electric.values/1000*prices))

if __name__=='__main__':
    from simulation import simulate
    from optimization import simulate_optimized
    p=tou_price(168)
    print('Thermostat cost:',round(cost(simulate(),p),2))
    print('Optimized cost:',round(cost(simulate_optimized(),p),2))
